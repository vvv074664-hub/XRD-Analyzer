import pandas as pd
import streamlit as st

from analyzer import XRDDataError, read_xrd_data

# 网页基本设置
st.set_page_config(
    page_title="XRD Analyzer",
    page_icon="📊",
    layout="wide"
)

# 网页标题
st.title("XRD Analyzer")
st.subheader("X射线衍射数据分析工具")

st.write(
    "欢迎使用 XRD Analyzer。"
    "本工具用于材料科学实验中的 XRD 数据读取、可视化与分析。"
)

st.divider()

# 上传数据
st.header("1. 上传 XRD 实验数据")

uploaded_file = st.file_uploader(
    "请选择 XRD 数据文件",
    type=["csv", "txt"]
)


def format_parameter(value, unit=""):
    """将已识别的参数转换为页面显示文本。"""
    if value is None:
        return "未识别"
    if isinstance(value, float):
        return f"{value:g}{unit}"
    return f"{value}{unit}"


# 读取并显示数据
if uploaded_file is not None:
    try:
        result = read_xrd_data(uploaded_file, uploaded_file.name)
        data = result.data
        metadata = result.metadata
    except XRDDataError as exc:
        st.error(f"文件读取失败：{exc}")
    except Exception as exc:
        st.error(f"文件读取失败：发生未预期的错误，请检查文件内容。详细信息：{exc}")
    else:
        st.success("文件读取成功，数据校验通过！")

        st.subheader("文件与数据概况")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("文件名称", uploaded_file.name)
        col2.metric("样品名称", metadata["sample_name"] or "未识别")
        col3.metric("数据点数量", len(data))
        col4.metric(
            "2θ 范围",
            f"{data['2theta'].min():g}° ~ {data['2theta'].max():g}°",
        )

        st.subheader("实验参数")
        parameters = [
            ("扫描起始角度", format_parameter(metadata["start_angle"], "°")),
            ("扫描结束角度", format_parameter(metadata["stop_angle"], "°")),
            ("扫描步长", format_parameter(metadata["step_size"], "°")),
            ("仪器型号", format_parameter(metadata["instrument"])),
            ("X 射线管电压", format_parameter(metadata["tube_voltage_kv"], " kV")),
            ("X 射线管电流", format_parameter(metadata["tube_current_ma"], " mA")),
        ]
        parameter_table = pd.DataFrame(parameters, columns=["参数", "数值"])
        st.dataframe(parameter_table, hide_index=True, use_container_width=True)

        st.subheader("数据预览")
        preview = data.head(100).rename(
            columns={"2theta": "2θ (°)", "Intensity": "Intensity"}
        )
        st.dataframe(preview, hide_index=True, use_container_width=True)
        if len(data) > 100:
            st.caption(f"仅显示前 100 行，完整数据共 {len(data)} 行。")

else:
    st.info("请先上传 CSV 或 TXT 格式的 XRD 数据文件。")
