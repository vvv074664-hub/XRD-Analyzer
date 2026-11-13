"""XRD 数据读取、校验与实验元信息提取。"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
NUMBER_RE = re.compile(NUMBER_PATTERN)
METADATA_RE = re.compile(
    r"^(Sample|Goniometer|Start|Stop|Step|X-Ray)\s*(?:\t+|:|=|\s+)\s*(.*)$",
    re.IGNORECASE,
)


class XRDDataError(ValueError):
    """数据文件无法读取或未通过校验。"""


@dataclass
class XRDData:
    """读取后的 XRD 数据及其实验元信息。"""

    data: pd.DataFrame
    metadata: dict[str, str | float | None]


def _empty_metadata() -> dict[str, str | float | None]:
    return {
        "sample_name": None,
        "start_angle": None,
        "stop_angle": None,
        "step_size": None,
        "instrument": None,
        "tube_voltage_kv": None,
        "tube_current_ma": None,
    }


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        try:
            return Path(source).read_bytes()
        except OSError as exc:
            raise XRDDataError(f"无法打开文件：{exc}") from exc

    try:
        if hasattr(source, "getvalue"):
            return source.getvalue()
        return source.read()
    except (OSError, ValueError) as exc:
        raise XRDDataError(f"无法读取上传的文件：{exc}") from exc


def _decode_text(raw: bytes) -> str:
    if not raw:
        raise XRDDataError("文件为空，请上传包含 XRD 数据的文件。")
    if b"\x00" in raw:
        raise XRDDataError("文件内容不是可识别的文本数据，请检查文件格式。")

    for encoding in ("utf-8-sig", "gb18030", "shift_jis", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise XRDDataError("无法识别文件编码，请将文件另存为 UTF-8 后重试。")


def _is_number(token: str) -> bool:
    return NUMBER_RE.fullmatch(token.strip()) is not None


def _metadata_number(value: str) -> float | None:
    match = NUMBER_RE.search(value)
    if match is None:
        return None
    number = float(match.group())
    return number if math.isfinite(number) else None


def _parse_metadata_line(
    line: str, metadata: dict[str, str | float | None]
) -> None:
    match = METADATA_RE.match(line.strip())
    if match is None:
        return

    key = match.group(1).lower()
    value = match.group(2).strip()
    if key == "sample":
        metadata["sample_name"] = value or None
    elif key == "goniometer":
        metadata["instrument"] = value or None
    elif key == "start":
        metadata["start_angle"] = _metadata_number(value)
    elif key == "stop":
        metadata["stop_angle"] = _metadata_number(value)
    elif key == "step":
        metadata["step_size"] = _metadata_number(value)
    elif key == "x-ray":
        voltage = re.search(rf"({NUMBER_PATTERN})\s*kV", value, re.IGNORECASE)
        current = re.search(rf"({NUMBER_PATTERN})\s*mA", value, re.IGNORECASE)
        if voltage:
            metadata["tube_voltage_kv"] = float(voltage.group(1))
        if current:
            metadata["tube_current_ma"] = float(current.group(1))


def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise XRDDataError("未找到有效的两列数值数据。")

    result = data.copy()
    for column in ("2theta", "Intensity"):
        original = result[column]
        numeric = pd.to_numeric(original, errors="coerce")
        missing = original.isna() | original.astype(str).str.strip().eq("")
        if missing.any():
            rows = (original.index[missing].tolist()[:5])
            raise XRDDataError(
                f"{column} 列存在空值，数据行位置：{rows}，请补全或删除无效行。"
            )

        non_numeric = numeric.isna()
        if non_numeric.any():
            rows = (original.index[non_numeric].tolist()[:5])
            raise XRDDataError(
                f"{column} 列包含非数值内容，数据行位置：{rows}。"
            )

        not_finite = ~numeric.map(math.isfinite)
        if not_finite.any():
            rows = (original.index[not_finite].tolist()[:5])
            raise XRDDataError(
                f"{column} 列包含无穷大等无效数值，数据行位置：{rows}。"
            )
        result[column] = numeric.astype(float)

    result = result.reset_index(drop=True)
    angles = result["2theta"].tolist()
    differences = [later - earlier for earlier, later in zip(angles, angles[1:])]
    if any(difference == 0 for difference in differences):
        raise XRDDataError("2θ 列存在重复角度，请检查数据是否重复或排列异常。")
    if differences and not (
        all(difference > 0 for difference in differences)
        or all(difference < 0 for difference in differences)
    ):
        raise XRDDataError("2θ 列必须按递增或递减顺序排列，不能无序或来回跳变。")

    return result[["2theta", "Intensity"]]


def _read_txt(text: str) -> XRDData:
    rows: list[tuple[float, float]] = []
    metadata = _empty_metadata()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        tokens = re.split(r"[\t ]+", line)
        first_is_number = _is_number(tokens[0])
        if first_is_number:
            if len(tokens) != 2 or not _is_number(tokens[1]):
                raise XRDDataError(
                    f"第 {line_number} 行格式异常：数值数据必须且只能包含 "
                    "2θ 和 Intensity 两列有效数值。"
                )
            rows.append((float(tokens[0]), float(tokens[1])))
        elif rows:
            raise XRDDataError(
                f"第 {line_number} 行格式异常：数值数据区中出现非数值内容。"
            )
        else:
            _parse_metadata_line(line, metadata)

    if not rows:
        raise XRDDataError(
            "未找到有效的两列数值数据，请确认文件包含 2θ 和 Intensity。"
        )

    data = pd.DataFrame(rows, columns=["2theta", "Intensity"])
    return XRDData(_validate_data(data), metadata)


def _normalise_column_name(column: object) -> str:
    return re.sub(r"[\s_\-()（）°/\\]+", "", str(column)).lower()


def _find_column(columns: pd.Index, aliases: set[str]) -> object | None:
    for column in columns:
        if _normalise_column_name(column) in aliases:
            return column
    return None


def _read_csv_frame(text: str, header: str | None = "infer") -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.StringIO(text), sep=None, engine="python", header=header
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise XRDDataError(f"CSV 格式异常，无法解析：{exc}") from exc


def _read_csv(text: str) -> XRDData:
    frame = _read_csv_frame(text)
    if frame.empty:
        raise XRDDataError("CSV 文件中没有可读取的数据。")

    angle_aliases = {"2theta", "2θ", "theta", "angle", "2thetadeg", "角度", "2θ角度"}
    intensity_aliases = {"intensity", "counts", "count", "cps", "强度", "计数"}
    angle_column = _find_column(frame.columns, angle_aliases)
    intensity_column = _find_column(frame.columns, intensity_aliases)

    # pd.read_csv 默认把第一行当表头；两个“表头”都是数值时按无表头文件重读。
    if (
        angle_column is None
        and intensity_column is None
        and len(frame.columns) == 2
        and all(_is_number(str(column)) for column in frame.columns)
    ):
        frame = _read_csv_frame(text, header=None)
        angle_column, intensity_column = frame.columns[:2]
    elif angle_column is None and intensity_column is None:
        if len(frame.columns) != 2:
            raise XRDDataError(
                "CSV 必须包含两列数据，或提供可识别的 2theta、Intensity 表头。"
            )
        angle_column, intensity_column = frame.columns[:2]
    elif angle_column is None or intensity_column is None:
        if len(frame.columns) != 2:
            raise XRDDataError("无法同时识别 CSV 中的 2θ 和 Intensity 两列。")
        remaining = [
            column
            for column in frame.columns
            if column not in (angle_column, intensity_column)
        ]
        if angle_column is None:
            angle_column = remaining[0]
        else:
            intensity_column = remaining[0]

    data = frame[[angle_column, intensity_column]].copy()
    data.columns = ["2theta", "Intensity"]
    return XRDData(_validate_data(data), _empty_metadata())


def read_xrd_data(
    source: str | Path | bytes | BinaryIO, filename: str | None = None
) -> XRDData:
    """读取 CSV/TXT XRD 文件，返回两列数值数据和实验元信息。

    source 可以是文件路径、字节内容或 Streamlit 上传的文件对象。
    """
    if filename is None:
        filename = getattr(source, "name", None)
        if filename is None and isinstance(source, (str, Path)):
            filename = str(source)
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise XRDDataError("仅支持 CSV 和 TXT 格式的 XRD 数据文件。")

    text = _decode_text(_read_bytes(source))
    if suffix == ".txt":
        return _read_txt(text)
    return _read_csv(text)
