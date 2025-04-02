import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, MaxPooling1D, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from mlxtend.regressor import StackingRegressor
import openpyxl
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import base64
from io import BytesIO
import os
from datetime import datetime
import json  # 用于导出JSON格式的图表数据
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import tempfile
from scipy import stats

# 尝试导入statsmodels，如果失败则给出警告
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# 设置页面布局
st.set_page_config(
    layout="wide", 
    page_title="机器学习模型集成分析系统",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置英文和中文字体
mpl.rcParams['savefig.dpi'] = 300  # 设置保存图片的DPI
mpl.rcParams['figure.dpi'] = 300  # 设置显示图片的DPI

# 标题
st.title("机器学习模型集成与可视化分析系统 📈")
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #1E88E5;
    }
    .subheader {
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
        color: #0277BD;
    }
    .divider {
        margin-top: 10px;
        margin-bottom: 10px;
        border-top: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# 语言选择
language = st.sidebar.selectbox("选择语言/Select Language", ["中文", "English"])

# 翻译字典
translations = {
    "中文": {
        "upload_data": "上传数据集 (优先Excel, 也支持CSV)",
        "show_data": "显示完整数据",
        "data_preprocessing": "数据预处理",
        "missing_value": "处理缺失值方法",
        "missing_options": ["删除缺失行", "均值填充", "中位数填充", "众数填充", "KNN填充", "插值填充"],
        "scaling": "特征缩放方法",
        "scaling_options": ["无", "标准化 (StandardScaler)", "归一化 (MinMaxScaler)", "稳健归一化 (RobustScaler)"],
        "data_split": "数据划分",
        "test_size": "测试集比例",
        "random_state": "随机种子",
        "target": "选择目标变量",
        "features": "选择特征变量",
        "data_exploration": "数据探索性分析",
        "stats": "数据统计信息",
        "correlation": "特征相关性分析",
        "distribution": "特征分布",
        "target_dist": "目标变量分布",
        "3d_plot": "3D特征关系图",
        "model_training": "模型训练",
        "select_models": "选择要训练的模型",
        "model_options": [
            "支持向量回归", "随机森林", "XGBoost", 
            "LightGBM", "CatBoost", "Stacking集成", 
            "贝叶斯岭回归", "多层感知机", "K近邻", "决策树"
        ],
        "start_training": "开始训练",
        "model_evaluation": "模型评估",
        "performance": "模型性能比较",
        "detailed_analysis": "模型详细分析",
        "visualization": "可视化分析",
        "model_saving": "模型保存与导出",
        "save_model": "保存模型",
        "export_results": "导出结果",
        "export_charts": "导出图表",
        "time_series": "时间序列分析",
        "export_all_charts": "导出所有图表",
        "export_format": "导出格式",
        "export_success": "图表已成功导出到 charts 文件夹!"
    },
    "English": {
        "upload_data": "Upload Dataset (Excel preferred, CSV also supported)",
        "show_data": "Show Full Data",
        "data_preprocessing": "Data Preprocessing",
        "missing_value": "Missing Value Handling",
        "missing_options": ["Drop missing rows", "Mean imputation", "Median imputation", "Mode imputation", "KNN imputation", "Interpolation"],
        "scaling": "Feature Scaling Method",
        "scaling_options": ["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)", "Robust Scaling"],
        "data_split": "Data Splitting",
        "test_size": "Test Size Ratio",
        "random_state": "Random State",
        "target": "Select Target Variable",
        "features": "Select Feature Variables",
        "data_exploration": "Data Exploration",
        "stats": "Data Statistics",
        "correlation": "Feature Correlation Analysis",
        "distribution": "Feature Distribution",
        "target_dist": "Target Variable Distribution",
        "3d_plot": "3D Feature Relationship",
        "model_training": "Model Training",
        "select_models": "Select Models to Train",
        "model_options": [
            "Support Vector Regression", "Random Forest", "XGBoost", 
            "LightGBM", "CatBoost", "Stacking Ensemble", 
            "Bayesian Ridge", "MLP", "K-Nearest Neighbors", "Decision Tree"
        ],
        "start_training": "Start Training",
        "model_evaluation": "Model Evaluation",
        "performance": "Model Performance Comparison",
        "detailed_analysis": "Detailed Model Analysis",
        "visualization": "Visualization Analysis",
        "model_saving": "Model Saving & Export",
        "save_model": "Save Model",
        "export_results": "Export Results",
        "export_charts": "Export Charts",
        "time_series": "Time Series Analysis",
        "export_all_charts": "Export All Charts",
        "export_format": "Export Format",
        "export_success": "Charts successfully exported to charts folder!"
    }
}

def t(key):
    """翻译函数"""
    return translations[language][key]

# 添加文件名清理函数
def clean_filename(filename):
    """清理文件名，替换非法字符"""
    # 替换常见的非法字符
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    cleaned_name = filename
    for char in illegal_chars:
        cleaned_name = cleaned_name.replace(char, '_')
    return cleaned_name

# 添加Excel工作表名称清理函数
def clean_sheet_name(sheet_name):
    """清理Excel工作表名称"""
    illegal_chars = ['/', '\\', '?', '*', '[', ']']
    cleaned_name = sheet_name
    for char in illegal_chars:
        cleaned_name = cleaned_name.replace(char, '_')
    # Excel工作表名称长度限制为31个字符
    return cleaned_name[:31]

# 添加save_chart函数
def save_chart(fig, filename, data=None, folder="charts", format="png", dpi=300):
    """
    保存图表到文件和对应的数据文件
    返回包含成功状态和文件路径的字典
    """
    # 确保目录存在
    os.makedirs(folder, exist_ok=True)
    result = {"success": False, "format": None, "filepath": None, "data_filepath": None}
    
    # 保存图表数据为CSV (用于Origin绘图)
    if data is not None:
        try:
            data_path = f"{folder}/{filename}_data.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(data_path, index=False)
            elif isinstance(data, dict):
                pd.DataFrame(data).to_csv(data_path, index=False)
            result["data_filepath"] = data_path
        except Exception as e:
            st.warning(f"保存图表数据失败: {str(e)}")
    
    # 保存图表图像
    try:
        if isinstance(fig, plt.Figure):
            # Matplotlib图表
            if format == "html":
                # 如果需要HTML格式，保存为PNG后转HTML
                filepath = os.path.join(folder, f"{filename}.png")
                fig.savefig(filepath, format="png", dpi=dpi, bbox_inches="tight")
                
                # 创建一个包含图像的简单HTML文件
                html_path = os.path.join(folder, f"{filename}.html")
                with open(html_path, "w") as f:
                    f.write(f"<html><body><img src='{os.path.basename(filepath)}' /></body></html>")
                
                result["success"] = True
                result["format"] = "html"
                result["filepath"] = html_path
            else:
                # 直接保存为要求的格式
                filepath = f"{folder}/{filename}.{format}"
                fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
                result["success"] = True
                result["format"] = format
                result["filepath"] = filepath
        else:
            # Plotly图表 - 直接保存为HTML，忽略其他格式
            html_path = f"{folder}/{filename}.html"
            fig.write_html(html_path)
            result["success"] = True
            result["format"] = "html"
            result["filepath"] = html_path
            
            # 如果用户原本想要其他格式，给出提示
            if format != "html":
                st.info(f"Plotly图表已保存为HTML格式。如需其他格式，请安装kaleido: pip install kaleido")
    except Exception as e:
        st.error(f"导出图表失败: {str(e)}")
    
    return result

# 初始化session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'language' not in st.session_state:
    st.session_state.language = language

# 在应用程序初始化部分添加
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []  # 初始化为空列表

# 侧边栏 - 数据上传和预处理
with st.sidebar:
    st.markdown('<p class="main-header">🛠️ 数据处理控制面板</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 上传数据
    uploaded_file = st.file_uploader(t("upload_data"), type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            
            st.session_state.data = data
            st.success("数据上传成功!" if language == "中文" else "Data uploaded successfully!")
            
            # 显示完整数据
            if st.checkbox(t("show_data")):
                st.dataframe(data)
                
            # 数据预处理选项
            st.subheader(t("data_preprocessing"))
            
            # 处理缺失值
            if data.isnull().sum().sum() > 0:
                st.warning("数据中存在缺失值!" if language == "中文" else "Missing values detected!")
                missing_options = ["删除缺失行", "均值填充", "中位数填充", "众数填充", "KNN填充", "插值填充"]
                missing_option = st.selectbox(t("missing_value"), missing_options, key="missing_value_selectbox_sidebar")
                
                processed_data = data.copy()
                
                if missing_option == "删除缺失行":
                    processed_data = processed_data.dropna()
                    st.info(f"已删除 {data.shape[0] - processed_data.shape[0]} 行含缺失值的数据")
                elif missing_option == "均值填充":
                    for col in processed_data.columns:
                        if processed_data[col].dtype.kind in 'fc':  # 只对数值列进行均值填充
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                    st.info("已使用均值填充数值列的缺失值")
                elif missing_option == "中位数填充":
                    for col in processed_data.columns:
                        if processed_data[col].dtype.kind in 'fc':  # 只对数值列进行中位数填充
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    st.info("已使用中位数填充数值列的缺失值")
                elif missing_option == "众数填充":
                    for col in processed_data.columns:
                        if processed_data[col].isnull().sum() > 0:  # 只处理有缺失值的列
                            if not processed_data[col].empty:
                                mode_val = processed_data[col].mode()
                                if not mode_val.empty:
                                    processed_data[col] = processed_data[col].fillna(mode_val[0])
                    st.info("已使用众数填充缺失值")
                elif missing_option == "KNN填充":
                    try:
                        from sklearn.impute import KNNImputer
                        # 先检查是否有非数值列
                        num_cols = processed_data.select_dtypes(include=['float', 'int']).columns
                        if len(num_cols) > 0:
                            imputer = KNNImputer(n_neighbors=5)
                            processed_data[num_cols] = pd.DataFrame(
                                imputer.fit_transform(processed_data[num_cols]),
                                columns=num_cols,
                                index=processed_data.index
                            )
                            st.info("已使用KNN方法填充数值列的缺失值")
                        else:
                            st.warning("数据中没有数值列，无法使用KNN填充")
                    except Exception as e:
                        st.error(f"KNN填充错误: {str(e)}")
                elif missing_option == "插值填充":
                    for col in processed_data.columns:
                        if processed_data[col].dtype.kind in 'fc':  # 只对数值列进行插值填充
                            processed_data[col] = processed_data[col].interpolate(method='linear').fillna(
                                method='bfill').fillna(method='ffill')  # 前向和后向填充处理边界缺失值
                    st.info("已使用插值方法填充数值列的缺失值")
                
                # 显示处理前后的缺失值情况
                missing_before = data.isnull().sum().sum()
                missing_after = processed_data.isnull().sum().sum()
                
                if missing_after > 0:
                    st.warning(f"处理后仍有 {missing_after} 个缺失值")
                else:
                    st.success("所有缺失值已处理完成!")
                
                # 更新数据
                data = processed_data
                st.session_state.data = processed_data
            
            # 特征缩放
            scale_option = st.selectbox(
                t("scaling"),
                t("scaling_options")
            )
            
            # 数据分割
            st.subheader(t("data_split"))
            test_size = st.slider(t("test_size"), 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
            random_state = st.number_input(t("random_state"), 0, 100, 42, key="random_state_input")
            
            # 选择特征和目标
            target_col = st.selectbox(t("target"), data.columns, key="target_select")
            feature_cols = st.multiselect(t("features"), [col for col in data.columns if col != target_col], key="features_select")
            
            if feature_cols and target_col:
                X = data[feature_cols]
                y = data[target_col]
                
                # 应用特征缩放
                if scale_option == t("scaling_options")[1]:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                elif scale_option == t("scaling_options")[2]:
                    scaler = MinMaxScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                elif scale_option == t("scaling_options")[3]:
                    scaler = RobustScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                
                # 分割数据集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                st.success(f"{'数据划分完成!' if language == '中文' else 'Data split completed!'} 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
                
                # 保存到session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_cols = feature_cols
                st.session_state.target_col = target_col
                st.session_state.scaler = scaler if scale_option != t("scaling_options")[0] else None
                
                # 添加预览训练集和测试集功能
                with st.expander("预览训练集和测试集"):
                    preview_tab1, preview_tab2 = st.tabs(["训练集", "测试集"])
                    
                    with preview_tab1:
                        # 创建完整训练集预览
                        train_df = pd.concat([X_train, y_train], axis=1)
                        
                        # 添加分页选项
                        train_page_size = st.slider("每页显示行数", 10, 100, 50, key="train_page_size")
                        train_total_pages = (train_df.shape[0] + train_page_size - 1) // train_page_size
                        
                        train_page = st.number_input(
                            f"页码 (共 {train_total_pages} 页)", 
                            min_value=1, 
                            max_value=train_total_pages,
                            value=1,
                            key="train_page"
                        )
                        
                        # 计算当前页的起始和结束索引
                        train_start_idx = (train_page - 1) * train_page_size
                        train_end_idx = min(train_start_idx + train_page_size, train_df.shape[0])
                        
                        # 显示当前页的数据
                        st.dataframe(train_df.iloc[train_start_idx:train_end_idx])
                        st.caption(f"显示 {train_start_idx+1} 到 {train_end_idx} 行 (共 {train_df.shape[0]} 行)")
                        
                        # 添加查看全部训练集数据的选项
                        if st.checkbox("显示全部训练集数据", key="show_all_train"):
                            st.dataframe(train_df)
                            st.caption(f"显示全部 {train_df.shape[0]} 行")
                        
                        # 添加统计摘要选项
                        if st.checkbox("显示训练集统计摘要", key="show_train_stats"):
                            st.write("**训练集统计摘要**")
                            st.dataframe(train_df.describe())
                        
                        # 添加训练集导出功能
                        st.subheader("导出训练集数据")
                        col_train1, col_train2 = st.columns(2)
                        
                        with col_train1:
                            train_filename = st.text_input("文件名", "train_data.csv", key="train_filename")
                        
                        with col_train2:
                            train_dir = st.text_input("保存目录", os.getcwd(), key="train_dir")
                        
                        # 组合完整路径
                        train_export_path = os.path.join(train_dir, train_filename)
                        
                        # 显示最终路径
                        st.info(f"将保存到: {train_export_path}")
                        
                        # 选择文件格式
                        train_format = st.selectbox(
                            "导出格式", 
                            ["CSV (.csv)", "Excel (.xlsx)", "JSON (.json)"],
                            key="train_format"
                        )
                        
                        if st.button("导出训练集", key="export_train_button"):
                            try:
                                # 提取文件扩展名
                                file_ext = train_filename.split('.')[-1].lower() if '.' in train_filename else ''
                                
                                # 确保目录存在
                                os.makedirs(os.path.dirname(train_export_path) if os.path.dirname(train_export_path) else '.', exist_ok=True)
                                
                                # 根据选择的格式导出
                                if train_format == "CSV (.csv)":
                                    # 确保文件名以.csv结尾
                                    if not train_export_path.lower().endswith('.csv'):
                                        train_export_path += '.csv'
                                    train_df.to_csv(train_export_path, index=False)
                                elif train_format == "Excel (.xlsx)":
                                    # 确保文件名以.xlsx结尾
                                    if not train_export_path.lower().endswith('.xlsx'):
                                        train_export_path += '.xlsx'
                                    # 检查是否安装了openpyxl
                                    try:
                                        import openpyxl
                                        train_df.to_excel(train_export_path, index=False, engine='openpyxl')
                                    except ImportError:
                                        st.error("导出Excel需要安装openpyxl库，请运行: pip install openpyxl")
                                elif train_format == "JSON (.json)":
                                    # 确保文件名以.json结尾
                                    if not train_export_path.lower().endswith('.json'):
                                        train_export_path += '.json'
                                    train_df.to_json(train_export_path, orient="records")
                                
                                st.success(f"训练集数据已成功导出至 {train_export_path}")
                            except Exception as e:
                                st.error(f"导出失败: {str(e)}")
                                st.info("如果是Excel格式导出错误，请确保已安装openpyxl: pip install openpyxl")
                                st.info("如果是文件路径错误，请确保指定的目录存在且有写入权限")
                    
                    with preview_tab2:
                        # 创建完整测试集预览
                        test_df = pd.concat([X_test, y_test], axis=1)
                        
                        # 添加分页选项
                        test_page_size = st.slider("每页显示行数", 10, 100, 50, key="test_page_size")
                        test_total_pages = (test_df.shape[0] + test_page_size - 1) // test_page_size
                        
                        test_page = st.number_input(
                            f"页码 (共 {test_total_pages} 页)", 
                            min_value=1, 
                            max_value=test_total_pages,
                            value=1,
                            key="test_page"
                        )
                        
                        # 计算当前页的起始和结束索引
                        test_start_idx = (test_page - 1) * test_page_size
                        test_end_idx = min(test_start_idx + test_page_size, test_df.shape[0])
                        
                        # 显示当前页的数据
                        st.dataframe(test_df.iloc[test_start_idx:test_end_idx])
                        st.caption(f"显示 {test_start_idx+1} 到 {test_end_idx} 行 (共 {test_df.shape[0]} 行)")
                        
                        # 添加查看全部数据的选项
                        if st.checkbox("显示全部测试集数据", key="show_all_test"):
                            st.dataframe(test_df)
                            st.caption(f"显示全部 {test_df.shape[0]} 行")
                        
                        # 添加统计摘要选项
                        if st.checkbox("显示测试集统计摘要", key="show_test_stats"):
                            st.write("**测试集统计摘要**")
                            st.dataframe(test_df.describe())
                        
                        # 添加测试集导出功能
                        st.subheader("导出测试集数据")
                        col_test1, col_test2 = st.columns(2)
                        
                        with col_test1:
                            test_filename = st.text_input("文件名", "test_data.csv", key="test_filename")
                        
                        with col_test2:
                            test_dir = st.text_input("保存目录", os.getcwd(), key="test_dir")
                        
                        # 组合完整路径
                        test_export_path = os.path.join(test_dir, test_filename)
                        
                        # 显示最终路径
                        st.info(f"将保存到: {test_export_path}")
                        
                        # 选择文件格式
                        test_format = st.selectbox(
                            "导出格式", 
                            ["CSV (.csv)", "Excel (.xlsx)", "JSON (.json)"],
                            key="test_format"
                        )
                        
                        if st.button("导出测试集", key="export_test_button"):
                            try:
                                # 提取文件扩展名
                                file_ext = test_filename.split('.')[-1].lower() if '.' in test_filename else ''
                                
                                # 确保目录存在
                                os.makedirs(os.path.dirname(test_export_path) if os.path.dirname(test_export_path) else '.', exist_ok=True)
                                
                                # 根据选择的格式导出
                                if test_format == "CSV (.csv)":
                                    # 确保文件名以.csv结尾
                                    if not test_export_path.lower().endswith('.csv'):
                                        test_export_path += '.csv'
                                    test_df.to_csv(test_export_path, index=False)
                                elif test_format == "Excel (.xlsx)":
                                    # 确保文件名以.xlsx结尾
                                    if not test_export_path.lower().endswith('.xlsx'):
                                        test_export_path += '.xlsx'
                                    # 检查是否安装了openpyxl
                                    try:
                                        import openpyxl
                                        test_df.to_excel(test_export_path, index=False, engine='openpyxl')
                                    except ImportError:
                                        st.error("导出Excel需要安装openpyxl库，请运行: pip install openpyxl")
                                elif test_format == "JSON (.json)":
                                    # 确保文件名以.json结尾
                                    if not test_export_path.lower().endswith('.json'):
                                        test_export_path += '.json'
                                    test_df.to_json(test_export_path, orient="records")
                                
                                st.success(f"测试集数据已成功导出至 {test_export_path}")
                            except Exception as e:
                                st.error(f"导出失败: {str(e)}")
                                st.info("如果是Excel格式导出错误，请确保已安装openpyxl: pip install openpyxl")
                                st.info("如果是文件路径错误，请确保指定的目录存在且有写入权限")
        
        except Exception as e:
            st.error(f"{'数据读取错误:' if language == '中文' else 'Error reading data:'} {str(e)}")

    # 在数据处理完成后添加
    if 'data' in st.session_state:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("📤 数据导出")
        
        # 添加文件格式选项
        export_format = st.selectbox(
            "导出文件格式", 
            ["CSV (.csv)", "Excel (.xlsx)", "JSON (.json)"], 
            key="export_data_format"
        )
        
        # 根据选择的格式提供默认文件名
        default_filename = {
            "CSV (.csv)": "processed_data.csv",
            "Excel (.xlsx)": "processed_data.xlsx",
            "JSON (.json)": "processed_data.json"
        }[export_format]
        
        col1, col2 = st.columns(2)
        with col1:
            export_filename = st.text_input("文件名", default_filename, key="export_filename")
        with col2:
            default_dir = os.getcwd()
            export_dir = st.text_input("保存目录", default_dir, key="export_dir")
        
        # 组合完整路径
        export_path = os.path.join(export_dir, export_filename)
        
        # 显示最终路径
        st.info(f"将保存到: {export_path}")
        
        # 添加高级选项
        with st.expander("高级导出选项"):
            include_index = st.checkbox("包含行索引", False, key="export_include_index")
            if export_format == "Excel (.xlsx)":
                sheet_name = st.text_input("工作表名称", "处理后数据", key="export_sheet_name")
            elif export_format == "CSV (.csv)":
                encoding = st.selectbox("文件编码", ["utf-8", "gbk", "latin1"], key="export_encoding")
                separator = st.selectbox("分隔符", [",", ";", "tab"], key="export_separator")
                if separator == "tab":
                    separator = "\t"
        
        if st.button("导出处理后的数据集", key="export_data_button"):
            try:
                # 根据选择的格式导出数据
                if export_format == "CSV (.csv)":
                    st.session_state.data.to_csv(
                        export_path, 
                        index=include_index, 
                        encoding=encoding, 
                        sep=separator
                    )
                elif export_format == "Excel (.xlsx)":
                    st.session_state.data.to_excel(
                        export_path, 
                        index=include_index, 
                        sheet_name=sheet_name
                    )
                elif export_format == "JSON (.json)":
                    st.session_state.data.to_json(
                        export_path, 
                        orient="records" if not include_index else "columns"
                    )
                
                # 成功提示
                st.success(f"数据集已成功导出为 {export_path}")
                
                # 添加下载按钮
                with open(export_path, "rb") as f:
                    st.download_button(
                        "下载导出的文件",
                        f,
                        file_name=os.path.basename(export_path),
                        mime={
                            "CSV (.csv)": "text/csv",
                            "Excel (.xlsx)": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "JSON (.json)": "application/json"
                        }[export_format]
                    )
            except Exception as e:
                st.error(f"导出数据失败: {str(e)}")

# 在侧边栏添加训练集和测试集上传选项
train_file = st.sidebar.file_uploader("上传训练集", type=["csv", "xlsx"])
test_file = st.sidebar.file_uploader("上传测试集", type=["csv", "xlsx"])

if train_file and test_file:
    train_data = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
    test_data = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
    st.session_state.train_data = train_data
    st.session_state.test_data = test_data

# 主界面
if 'X_train' in st.session_state:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t("data_exploration"), t("model_training"), t("model_evaluation"), 
        t("visualization"), t("model_saving"), "模型预测"  # 添加新的标签页
    ])
    
    with tab1:
        st.header(t("data_exploration"))
        
        # 统计信息部分保持在上方
        st.subheader(t("stats"))  # 移除多余的缩进
        st.dataframe(st.session_state.X_train.describe())
        
        # 重新布局相关性分析和特征分布图
        col1, col2 = st.columns(2)
        
        with col1:
            # 相关性分析
            st.subheader(t("correlation"))
            corr_matrix = pd.concat([st.session_state.X_train, st.session_state.y_train], axis=1).corr()
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            
            # 显示相关性热图
            st.pyplot(fig_corr)
            
            # 添加查看大图和导出功能
            with st.expander("导出数据"):
                st.pyplot(fig_corr, use_container_width=True)
                
                # 相关性矩阵导出功能
                st.subheader("导出相关性矩阵数据")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    corr_filename = st.text_input("文件名", "correlation_matrix.csv", key="corr_filename")
                
                with col_b:
                    default_dir = os.getcwd()
                    save_dir = st.text_input("保存目录", default_dir, key="corr_save_dir")
                
                # 组合完整路径
                corr_export_path = os.path.join(save_dir, corr_filename)
                
                # 显示最终路径
                st.info(f"将保存到: {corr_export_path}")
                
                if st.button("导出相关性矩阵", key="export_corr_button"):
                    try:
                        # 确保目录存在
                        os.makedirs(os.path.dirname(corr_export_path), exist_ok=True)
                        # 导出为CSV
                        corr_matrix.to_csv(corr_export_path)
                        st.success(f"相关性矩阵已成功导出至 {corr_export_path}")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
                
                # 添加提示信息
                st.info("导出的相关性矩阵可以在Origin中重新绘制热图")
            
        with col2:
            # 特征分布
            st.subheader(t("distribution"))
            selected_feature = st.selectbox(t("distribution"), st.session_state.feature_cols, key="feature_dist_select")
            
            fig_dist, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.X_train[selected_feature], kde=True, ax=ax)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency' if language == "English" else '频率')
            st.pyplot(fig_dist)
            
            # 添加特征分布数据导出功能
            with st.expander("导出特征分布数据"):
                dist_filename = st.text_input("文件名", f"{selected_feature}_distribution.csv", key="dist_filename")
                dist_dir = st.text_input("保存目录", os.getcwd(), key="dist_dir")
                
                # 组合完整路径
                dist_export_path = os.path.join(dist_dir, dist_filename)
                
                # 显示最终路径
                st.info(f"将保存到: {dist_export_path}")
                
                if st.button("导出分布数据", key="export_dist_button"):
                    try:
                        # 准备导出数据
                        dist_data = pd.DataFrame({
                            selected_feature: st.session_state.X_train[selected_feature].values
                        })
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(dist_export_path), exist_ok=True)
                        # 导出为CSV
                        dist_data.to_csv(dist_export_path, index=False)
                        st.success(f"特征分布数据已成功导出至 {dist_export_path}")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
        
        # 目标变量分布
        st.subheader(t("target_dist"))
        fig_target, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(st.session_state.y_train, kde=True, ax=ax)  # 修正缩进
        ax.set_xlabel(st.session_state.target_col)
        ax.set_ylabel('Frequency' if language == "English" else '频率')
        st.pyplot(fig_target)
        
        # 添加目标变量分布数据导出功能
        with st.expander("导出目标变量分布数据"):
            target_filename = st.text_input("文件名", f"{st.session_state.target_col}_distribution.csv", key="target_filename")
            target_dir = st.text_input("保存目录", os.getcwd(), key="target_dir")
            
            # 组合完整路径
            target_export_path = os.path.join(target_dir, target_filename)
            
            # 显示最终路径
            st.info(f"将保存到: {target_export_path}")
            
            if st.button("导出目标变量数据", key="export_target_button"):
                try:
                    # 准备导出数据
                    target_data = pd.DataFrame({
                        st.session_state.target_col: st.session_state.y_train.values
                    })
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(target_export_path), exist_ok=True)
                    # 导出为CSV
                    target_data.to_csv(target_export_path, index=False)
                    st.success(f"目标变量分布数据已成功导出至 {target_export_path}")
                except Exception as e:
                    st.error(f"导出失败: {str(e)}")
            
            # 3D散点图（如果特征数量>=2）
            if len(st.session_state.feature_cols) >= 2:
                st.subheader(t("3d_plot"))
            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
                
            # 3D图的下拉选择框改为水平布局
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                x_feature = st.selectbox("X轴特征" if language == "中文" else "X Feature", 
                                       st.session_state.feature_cols, index=0, key="x_feature_select")
            with col_y:
                y_feature = st.selectbox("Y轴特征" if language == "中文" else "Y Feature", 
                                       st.session_state.feature_cols, index=1, key="y_feature_select")
                
            with col_z:
                if len(st.session_state.feature_cols) > 2:
                    z_feature = st.selectbox("Z轴特征" if language == "中文" else "Z Feature", 
                                           st.session_state.feature_cols, index=2, key="z_feature_select")
                else:
                    z_feature = y_feature
                
            # 绘制3D散点图
                ax.scatter(
                    st.session_state.X_train[x_feature],
                    st.session_state.X_train[y_feature],
                    st.session_state.X_train[z_feature],
                    c=st.session_state.y_train,
                    cmap='viridis'
                )
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                ax.set_zlabel(z_feature)
            st.pyplot(fig_3d)
            
            # 添加3D特征关系图数据导出功能
            st.subheader("导出3D特征关系数据")
            export_3d_container = st.container()

            with export_3d_container:
                plot3d_filename = st.text_input("文件名", f"3d_plot_{x_feature}_{y_feature}_{z_feature}.csv", key="plot3d_filename")
                plot3d_dir = st.text_input("保存目录", os.getcwd(), key="plot3d_dir")
                
                # 组合完整路径
                plot3d_export_path = os.path.join(plot3d_dir, plot3d_filename)
                
                # 显示最终路径
                st.info(f"将保存到: {plot3d_export_path}")
                
                if st.button("导出3D图数据", key="export_3d_button"):
                    try:
                        # 准备导出数据
                        plot3d_data = pd.DataFrame({
                            x_feature: st.session_state.X_train[x_feature].values,
                            y_feature: st.session_state.X_train[y_feature].values,
                            z_feature: st.session_state.X_train[z_feature].values,
                            st.session_state.target_col: st.session_state.y_train.values
                        })
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(plot3d_export_path), exist_ok=True)
                        # 导出为CSV
                        plot3d_data.to_csv(plot3d_export_path, index=False)
                        st.success(f"3D特征关系数据已成功导出至 {plot3d_export_path}")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
    
    with tab2:
        st.header(t("model_training"))
        
        # 将模型选择部分放在最前面
        st.subheader("选择要训练的模型")
        model_options = st.multiselect(
            t("select_models"),
            t("model_options"),
            default=[t("model_options")[0], t("model_options")[2]],
            key="model_select_tab2"
        )
        
        if model_options:  # 只有在选择了模型后才显示参数优化选项
            # 添加参数优化选择框
            st.subheader("参数优化设置")
            use_parameter_optimization = st.checkbox("使用参数优化", value=True, key="use_parameter_optimization_tab2")

            if use_parameter_optimization:
                # 使用 selectbox 替代 radio，并添加"不使用参数优化"选项
                optimization_mode = st.selectbox(
                    "参数优化模式",
                    ["自动优化参数", "不使用参数优化"],  
                    index=0,
                    key="optimization_mode_tab2"
                )
            
                if optimization_mode == "自动优化参数":
                    st.subheader("优化算法设置")
                    
                    opt_tabs = st.tabs(["传统机器学习优化"])
                    
                    with opt_tabs[0]:  # 传统机器学习优化
                        ml_optimization_method = st.selectbox(
                            "传统机器学习优化算法",
                            ["网格搜索 (GridSearchCV)", 
                             "随机搜索 (RandomizedSearchCV)", 
                             "贝叶斯优化 (BayesianOptimization)",
                             "遗传算法 (GeneticAlgorithm)",
                             "粒子群优化 (ParticleSwarmOptimization)",
                             "差分进化 (DifferentialEvolution)"],
                            key="ml_optimization_method_tab2"
                        )
                        
                        # 优化共通设置
                        ml_cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="ml_cv_folds_tab2")
                        ml_scoring_metric = st.selectbox(
                            "优化评估指标",
                            ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"],
                            key="ml_scoring_metric_tab2"
                        )
                        
                        # 各算法特定设置
                        if ml_optimization_method == "网格搜索 (GridSearchCV)":
                            ml_n_jobs = st.slider("并行任务数", -1, 8, -1, key="ml_grid_n_jobs_tab2")
                            st.info("并行任务数为-1表示使用所有可用CPU")
                            
                        elif ml_optimization_method == "随机搜索 (RandomizedSearchCV)":
                            ml_n_iter = st.slider("搜索迭代次数", 10, 200, 50, key="ml_random_n_iter_tab2")
                            ml_n_jobs = st.slider("并行任务数", -1, 8, -1, key="ml_random_n_jobs_tab2")
                            
                        elif ml_optimization_method == "贝叶斯优化 (BayesianOptimization)":
                            ml_init_points = st.slider("初始点数量", 2, 20, 5, key="ml_bayes_init_points_tab2")
                            ml_n_iter = st.slider("优化迭代次数", 5, 100, 25, key="ml_bayes_n_iter_tab2")
                            
                        elif ml_optimization_method == "遗传算法 (GeneticAlgorithm)":
                            ml_population_size = st.slider("种群大小", 10, 100, 30, key="ml_ga_population_tab2")
                            ml_generations = st.slider("迭代代数", 5, 50, 15, key="ml_ga_generations_tab2")
                            ml_mutation_rate = st.slider("变异率", 0.01, 0.5, 0.1, key="ml_ga_mutation_rate_tab2")
                            
                        elif ml_optimization_method == "粒子群优化 (ParticleSwarmOptimization)":
                            ml_n_particles = st.slider("粒子数量", 5, 50, 20, key="ml_pso_particles_tab2")
                            ml_n_iter = st.slider("迭代次数", 5, 50, 15, key="ml_pso_iterations_tab2")
                            
                        elif ml_optimization_method == "差分进化 (DifferentialEvolution)":
                            ml_population_size = st.slider("种群大小", 10, 100, 30, key="ml_de_population_tab2")
                            ml_generations = st.slider("迭代代数", 5, 50, 15, key="ml_de_generations_tab2")
                        else:
                            st.info("未使用参数优化，将使用默认参数训练模型。")
                else:
                    st.info("未使用参数优化，将使用默认参数训练模型。")
                    # 如果不使用参数优化，初始化变量以避免未定义错误
                    optimization_mode = "不使用参数优化"

            # 训练按钮放在底部
            if st.button(t("start_training"), key="train_button_tab2"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                X_train = st.session_state.X_train.values
                X_test = st.session_state.X_test.values
                y_train = st.session_state.y_train.values
                y_test = st.session_state.y_test.values
                
                models = {}
                results = {}
                
                for i, model_name in enumerate(model_options):
                    progress = (i + 1) / len(model_options)
                    progress_bar.progress(progress)
                    
                    status_text.text(f"正在训练 {model_name}...")
                    
                    start_time = time.time()
                    
                    try:
                        if use_parameter_optimization and optimization_mode == "自动优化参数":
                            # 使用参数优化的训练逻辑
                            # ... (保留原有的参数优化逻辑)
                            pass  # 添加pass语句防止缩进错误
                        else:
                            # 不使用参数优化，直接使用默认参数训练模型
                            if model_name == t("model_options")[0]:  # 支持向量回归
                                model = SVR()
                            elif model_name == t("model_options")[1]:  # 随机森林
                                model = RandomForestRegressor(random_state=random_state)
                            elif model_name == t("model_options")[2]:  # XGBoost
                                model = XGBRegressor(random_state=random_state)
                            elif model_name == t("model_options")[3]:  # LightGBM
                                model = LGBMRegressor(random_state=random_state)
                            elif model_name == t("model_options")[4]:  # CatBoost
                                model = CatBoostRegressor(random_state=random_state, verbose=0)
                            elif model_name == t("model_options")[6]:  # 贝叶斯岭回归
                                model = BayesianRidge()
                            elif model_name == t("model_options")[7]:  # 多层感知机
                                model = MLPRegressor(random_state=random_state, max_iter=1000)
                            elif model_name == t("model_options")[8]:  # K近邻
                                model = KNeighborsRegressor(n_neighbors=5)
                            elif model_name == t("model_options")[9]:  # 决策树
                                model = DecisionTreeRegressor(random_state=random_state)
                            elif model_name == t("model_options")[5]:  # Stacking集成
                                # 创建基础模型
                                base_models = [
                                    ('svr', SVR()),
                                    ('rf', RandomForestRegressor(random_state=random_state, n_estimators=100)),
                                    ('xgb', XGBRegressor(random_state=random_state))
                                ]
                                # 创建meta模型
                                meta_model = LinearRegression()
                                # 创建stacking模型
                                model = StackingRegressor(regressors=[model for _, model in base_models], meta_regressor=meta_model)
                            
                            model.fit(X_train, y_train)
                        
                        # 保存模型和结果
                        models[model_name] = model
                        y_pred = model.predict(X_test)
                        y_true = y_test
                        
                        # 计算评估指标
                        mse = mean_squared_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        
                        training_time = time.time() - start_time
                        
                        # 存储结果
                        results[model_name] = {
                            'mse': mse,
                            'r2': r2,
                            'mae': mae,
                            'training_time': training_time,
                            '极值': y_pred,
                            'y_true': y_true,
                            'use_parameter_optimization': use_parameter_optimization
                        }
                        
                    except Exception as e:
                        st.error(f"模型 {model_name} 训练或预测评估时出错: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        
                        # 创建一个空结果记录，以便用户知道模型已训练但评估失败
                        results[model_name] = {
                            'error': str(e),
                            'training_time': time.time() - start_time,
                            'use_parameter_optimization': use_parameter_optimization
                        }
                    
                    # 保存模型和结果到session state
                    st.session_state.models = models
                    st.session_state.results = results
                    
                    progress_bar.empty()
                    status_text.text("所有模型训练完成!")
        else:
            st.info("请先选择至少一个要训练的模型")
    
    with tab3:
        st.header(t("model_evaluation"))
        
        if 'models' in st.session_state and st.session_state.models:
            # 创建评估结果表格
            evaluation_data = []
            for model_name, result in st.session_state.results.items():
                if 'error' not in result:  # 只显示成功训练的模型
                    evaluation_data.append({
                        '模型名称': model_name,
                        'MSE': result['mse'],
                        'R²': result['r2'],
                        'MAE': result['mae'],
                        '训练时间(秒)': result['training_time']
                    })
            
            if evaluation_data:
                eval_df = pd.DataFrame(evaluation_data)
                # 按照R²值降序排序（精度从高到低）
                eval_df = eval_df.sort_values(by='R²', ascending=False)
                
                st.subheader("模型性能比较")
                st.dataframe(eval_df.style.highlight_min(subset=['MSE', 'MAE'], color='#a8d9a8')
                                      .highlight_max(subset=['R²'], color='#a8d9a8')
                                      .format({
                                          'MSE': '{:.6f}',
                                          'R²': '{:.6f}',
                                          'MAE': '{:.6f}',
                                          '训练时间(秒)': '{:.2f}'
                                      }))

                # 绘制性能比较图
                st.subheader("性能指标可视化")
                chart_type = st.selectbox("选择图表类型", 
                                         ["条形图", "雷达图"], key="chart_type_select")
                
                if chart_type == "条形图":
                    metric = st.selectbox("选择评估指标", ["MSE", "R²", "MAE"], key="metric_select")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(eval_df['模型名称'], eval_df[metric])
                    
                    # 改进数据标签显示
                    for bar in bars:
                        height = bar.get_height()
                        # 使用数学文本格式显示上标
                        if metric == "R²":
                            metric_label = "$\mathrm{R^2}$"
                        else:
                            metric_label = metric
                        
                        ax.annotate(f'{height:.4f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    rotation=0,
                                    fontsize=10,
                                    fontfamily='Times New Roman')
                    
                    # 改进图表样式
                    plt.xticks(rotation=45, ha='right', fontfamily='SimHei')
                    plt.title(f'模型{metric_label}比较', fontsize=12, fontfamily='SimHei')
                    plt.xlabel('模型', fontsize=10, fontfamily='SimHei')
                    plt.ylabel(metric_label, fontsize=10, fontfamily='Times New Roman')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 显示图表
                    st.pyplot(fig)
                    
                    # 添加数据导出功能
                    with st.expander("导出图表数据"):
                        # 准备导出数据
                        export_data = pd.DataFrame({
                            '模型': eval_df['模型名称'],
                            metric: eval_df[metric]
                        })
                        
                        # 提供多种导出格式选项
                        export_format = st.selectbox("选择导出格式", 
                                                   ["CSV", "Excel", "Origin格式(txt)"], 
                                                   key="bar_export_format")
                        
                        if st.button("导出数据", key="bar_export_button"):
                            try:
                                # 创建导出目录
                                os.makedirs("exported_data", exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                if export_format == "CSV":
                                    filepath = f"exported_data/bar_chart_data_{timestamp}.csv"
                                    # 使用 utf-8-sig 编码解决中文乱码
                                    export_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif export_format == "Excel":
                                    filepath = f"exported_data/bar_chart_data_{timestamp}.xlsx"
                                    export_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/bar_chart_data_{timestamp}.txt"
                                    # 使用 utf-8-sig 编码并添加制表符分隔
                                    export_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                # 提供下载按钮
                                with open(filepath, 'rb') as f:
                                    st.download_button(
                                        label="下载数据文件",
                                        data=f,
                                        file_name=os.path.basename(filepath),
                                        mime="text/csv" if export_format == "CSV" else 
                                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format == "Excel" else
                                             "text/plain"
                                    )
                                
                                st.success(f"数据已导出到: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")

                elif chart_type == "雷达图":
                    # 雷达图需要所有指标
                    model_names = eval_df['模型名称'].tolist()
                    mse_values = eval_df['MSE'].tolist()
                    r2_values = eval_df['R²'].tolist()
                    mae_values = eval_df['MAE'].tolist()
                    
                    # 归一化指标(因为MSE和MAE是越小越好，R²是越大越好)
                    max_mse = max(mse_values)
                    max_mae = max(mae_values)
                    
                    # 转换成统一方向（值越大越好）
                    norm_mse = [1 - (val / max_mse) for val in mse_values]
                    norm_r2 = r2_values  # R²已经是越大越好
                    norm_mae = [1 - (val / max_mae) for val in mae_values]
                    
                    # 绘制雷达图
                    categories = ['MSE表现', 'R²表现', 'MAE表现']
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, polar=True)
                    
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]  # 闭合雷达图
                    
                    for i, model_name in enumerate(model_names):
                        values = [norm_mse[i], norm_r2[i], norm_mae[i]]
                        values += values[:1]  # 闭合雷达图
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                        ax.fill(angles, values, alpha=0.1)
                    
                    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
                    ax.set_ylim(0, 1)
                    ax.grid(True)
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    st.pyplot(fig)
                    
                    # 添加数据导出功能
                    with st.expander("导出雷达图数据"):
                        # 准备导出数据
                        radar_export_data = pd.DataFrame({
                            '模型': model_names,
                            'MSE表现': norm_mse,
                            'R²表现': norm_r2,
                            'MAE表现': norm_mae
                        })
                        
                        export_format = st.selectbox("选择导出格式", 
                                                   ["CSV", "Excel", "Origin格式(txt)"], 
                                                   key="radar_export_format")
                        
                        if st.button("导出数据", key="radar_export_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                if export_format == "CSV":
                                    filepath = f"exported_data/radar_chart_data_{timestamp}.csv"
                                    radar_export_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif export_format == "Excel":
                                    filepath = f"exported_data/radar_chart_data_{timestamp}.xlsx"
                                    radar_export_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/radar_chart_data_{timestamp}.txt"
                                    radar_export_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                # 提供下载按钮
                                with open(filepath, 'rb') as f:
                                    st.download_button(
                                        label="下载数据文件",
                                        data=f,
                                        file_name=os.path.basename(filepath),
                                        mime="text/csv" if export_format == "CSV" else 
                                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format == "Excel" else
                                             "text/plain"
                                    )
                                
                                st.success(f"数据已导出到: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")
                
                # 检查NaN值
                if 'results_df' in locals() and 'nan_count' in locals():
                    if nan_count > 0:
                        with st.expander("查看包含NaN的数据"):
                            nan_rows = results_df[results_df['真实值'].isna() | results_df['预测值'].isna()]
                            st.write(f"包含NaN的行数: {len(nan_rows)}")
                            st.dataframe(nan_rows)

            # 如果没有模型
            if not st.session_state.models:
                st.warning("没有可用的评估结果。请先训练模型。")
        else:
            st.warning("请先训练模型。")
    
    with tab4:
        st.header(t("visualization"))
        
        if 'models' in st.session_state and st.session_state.models:
            # 创建多个可视化选项卡
            viz_tabs = st.tabs(["预测分析", "模型比较", "特征重要性", "残差分析"])
            
            with viz_tabs[0]:  # 预测分析
                st.subheader("预测结果可视化")
                model_select = st.selectbox("选择模型", list(st.session_state.models.keys()), key="model_select_viz1")
                
                if model_select:
                    result = st.session_state.results[model_select]
                    
                    # 预测vs真实值图表
                    fig = px.scatter(x=result['y_true'], y=result['极值'], 
                                    labels={'x': '真实值', 'y': '预测值'},
                                    title=f"{model_select}的预测结果")
                    
                    fig.add_shape(
                        type='line',
                        x0=min(result['y_true']),
                        y0=min(result['y_true']),
                        x1=max(result['y_true']),
                        y1=max(result['y_true']),
                        line=dict(color='red', width=2, dash='dash')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 添加预测结果数据导出功能
                    with st.expander("导出预测结果数据"):
                        pred_data = pd.DataFrame({
                            '真实值': result['y_true'],
                            '预测值': result['极值']
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            pred_filename = st.text_input("文件名", f"{model_select}_预测结果_{datetime.now().strftime('%Y%m%d')}.csv", 
                                                       key="pred_filename")
                        with col2:
                            pred_format = st.selectbox("导出格式", 
                                                     ["CSV", "Excel", "Origin格式(txt)"],
                                                     key="pred_format")
                        
                        if st.button("导出预测结果数据", key="export_pred_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                if pred_format == "CSV":
                                    filepath = f"exported_data/{pred_filename}"
                                    if not filepath.lower().endswith('.csv'):
                                        filepath += '.csv'
                                    pred_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif pred_format == "Excel":
                                    filepath = f"exported_data/{pred_filename}"
                                    if not filepath.lower().endswith('.xlsx'):
                                        filepath += '.xlsx'
                                    pred_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/{pred_filename}"
                                    if not filepath.lower().endswith('.txt'):
                                        filepath += '.txt'
                                    pred_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                st.success(f"数据已导出至: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")
                    
                    # 误差分析图表
                    errors = result['y_true'] - result['极值']
                    fig2 = px.scatter(x=result['极值'], y=errors,
                                     labels={'x': '预测值', 'y': '误差'},
                                     title=f"{model_select}的预测误差分析")
                    
                    fig2.add_shape(
                        type='line',
                        x0=min(result['极值']),
                        y0=0,
                        x1=max(result['极值']),
                        y1=0,
                        line=dict(color='red', width=2, dash='dash')
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 添加误差分析数据导出功能
                    with st.expander("导出误差分析数据"):
                        error_data = pd.DataFrame({
                            '预测值': result['极值'],
                            '误差': errors
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            error_filename = st.text_input("文件名", f"{model_select}_误差分析_{datetime.now().strftime('%Y%m%d')}.csv", 
                                                        key="error_filename")
                        with col2:
                            error_format = st.selectbox("导出格式", 
                                                      ["CSV", "Excel", "Origin格式(txt)"],
                                                      key="error_format")
                        
                        if st.button("导出误差分析数据", key="export_error_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                if error_format == "CSV":
                                    filepath = f"exported_data/{error_filename}"
                                    if not filepath.lower().endswith('.csv'):
                                        filepath += '.csv'
                                    error_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif error_format == "Excel":
                                    filepath = f"exported_data/{error_filename}"
                                    if not filepath.lower().endswith('.xlsx'):
                                        filepath += '.xlsx'
                                    error_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/{error_filename}"
                                    if not filepath.lower().endswith('.txt'):
                                        filepath += '.txt'
                                    error_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                st.success(f"数据已导出至: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")

            with viz_tabs[1]:  # 模型比较
                st.subheader("模型性能比较可视化")
                
                # 创建性能比较数据
                model_names = []
                mse_values = []
                r2_values = []
                mae_values = []
                
                for model_name, result in st.session_state.results.items():
                    if 'error' not in result:  # 只包含成功训练的模型
                        model_names.append(model_name)
                        mse_values.append(result['mse'])
                        r2_values.append(result['r2'])
                        mae_values.append(result['mae'])
                
                if model_names:
                    # 选择可视化类型
                    viz_type = st.selectbox("选择可视化类型", 
                                           ["条形图", "雷达图", "平行坐标图"], key="viz_type_select")
                    
                    if viz_type == "条形图":
                        metric = st.selectbox("选择评估指标", ["MSE", "R²", "MAE"], key="metric_viz_select")
                        
                        if metric == "MSE":
                            values = mse_values
                        elif metric == "R²":
                            values = r2_values
                        else:  # MAE
                            values = mae_values
                        
                        fig = px.bar(x=model_names, y=values, 
                                     labels={'x': '模型', 'y': metric},
                                     title=f"模型{metric}比较")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "雷达图":
                        # 雷达图需要归一化数据
                        max_mse = max(mse_values)
                        max_mae = max(mae_values)
                        
                        # 转换成统一方向（值越大越好）
                        norm_mse = [1 - (val / max_mse) for val in mse_values]
                        norm_r2 = r2_values  # R²已经是越大越好
                        norm_mae = [1 - (val / max_mae) for val in mae_values]
                        
                        # 创建雷达图数据
                        radar_data = []
                        for i, model in enumerate(model_names):
                            radar_data.append({
                                'Model': model,
                                'MSE表现': norm_mse[i],
                                'R²表现': norm_r2[i],
                                'MAE表现': norm_mae[i]
                            })
                        
                        fig = px.line_polar(pd.DataFrame(radar_data), r=radar_data[0].keys()[1:],
                                           theta=radar_data[0].keys()[1:], line_close=True,
                                           color='Model', range_r=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "平行坐标图":
                        # 准备平行坐标图数据
                        parallel_data = []
                        for i, model in enumerate(model_names):
                            parallel_data.append({
                                'Model': model,
                                'MSE': mse_values[i],
                                'R²': r2_values[i],
                                'MAE': mae_values[i]
                            })
                        
                        fig = px.parallel_coordinates(pd.DataFrame(parallel_data), color="Model",
                                                     dimensions=['MSE', 'R²', 'MAE'],
                                                     title="模型性能平行坐标图")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 添加模型比较数据导出功能
                    with st.expander("导出模型比较数据"):
                        comp_data = pd.DataFrame({
                            '模型': model_names,
                            'MSE': mse_values,
                            'R²': r2_values,
                            'MAE': mae_values
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            comp_filename = st.text_input("文件名", f"模型比较_{datetime.now().strftime('%Y%m%d')}.csv", 
                                                        key="comp_filename")
                        with col2:
                            comp_format = st.selectbox("导出格式", 
                                                     ["CSV", "Excel", "Origin格式(txt)"],
                                                     key="comp_format")
                        
                        if st.button("导出模型比较数据", key="export_comp_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                if comp_format == "CSV":
                                    filepath = f"exported_data/{comp_filename}"
                                    if not filepath.lower().endswith('.csv'):
                                        filepath += '.csv'
                                    comp_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif comp_format == "Excel":
                                    filepath = f"exported_data/{comp_filename}"
                                    if not filepath.lower().endswith('.xlsx'):
                                        filepath += '.xlsx'
                                    comp_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/{comp_filename}"
                                    if not filepath.lower().endswith('.txt'):
                                        filepath += '.txt'
                                    comp_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                st.success(f"数据已导出至: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")

            with viz_tabs[2]:  # 特征重要性
                st.subheader("特征重要性分析")
                
                feature_importance_models = []
                for model_name, model in st.session_state.models.items():
                    # 检查模型是否有feature_importances_属性
                    if hasattr(model, 'feature_importances_'):
                        feature_importance_models.append(model_name)
                
                if feature_importance_models:
                    model_for_importance = st.selectbox("选择模型", feature_importance_models, key="model_importance_select")
                    
                    if model_for_importance:
                        model = st.session_state.models[model_for_importance]
                        importances = model.feature_importances_
                        feature_names = st.session_state.feature_cols
                        
                        # 创建特征重要性数据
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='feature', y='importance',
                                     title=f"{model_for_importance}的特征重要性",
                                     labels={'importance': '重要性', 'feature': '特征'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示特征重要性数据表
                        st.write("特征重要性表")
                        st.dataframe(importance_df)
                    
                    # 添加特征重要性数据导出功能
                    with st.expander("导出特征重要性数据"):
                        imp_data = pd.DataFrame({
                            '特征': st.session_state.feature_cols,
                            '重要性': [model.feature_importances_[st.session_state.feature_cols.index(feature)] for feature in st.session_state.feature_cols]
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            imp_filename = st.text_input("文件名", f"{model_for_importance}_特征重要性_{datetime.now().strftime('%Y%m%d')}.csv", 
                                                       key="imp_filename")
                        with col2:
                            imp_format = st.selectbox("导出格式", 
                                                     ["CSV", "Excel", "Origin格式(txt)"],
                                                     key="imp_format")
                        
                        if st.button("导出特征重要性数据", key="export_imp_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                if imp_format == "CSV":
                                    filepath = f"exported_data/{imp_filename}"
                                    if not filepath.lower().endswith('.csv'):
                                        filepath += '.csv'
                                    imp_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                                elif imp_format == "Excel":
                                    filepath = f"exported_data/{imp_filename}"
                                    if not filepath.lower().endswith('.xlsx'):
                                        filepath += '.xlsx'
                                    imp_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/{imp_filename}"
                                    if not filepath.lower().endswith('.txt'):
                                        filepath += '.txt'
                                    imp_data.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
                                
                                st.success(f"数据已导出至: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")

            with viz_tabs[3]:  # 残差分析
                st.subheader("残差分析")
                model_for_residuals = st.selectbox("选择模型", list(st.session_state.models.keys()), key="model_residuals_select")
                
                if model_for_residuals:
                    result = st.session_state.results[model_for_residuals]
                    
                    # 计算残差
                    residuals = result['y_true'] - result['极值']
                    
                    # 创建残差图表
                    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # 残差散点图
                    ax1.scatter(result['极值'], residuals)
                    ax1.axhline(y=0, color='r', linestyle='--')
                    ax1.set_xlabel('预测值')
                    ax1.set_ylabel('残差')
                    ax1.set_title('残差 vs 预测值')
                    
                    # 残差直方图
                    ax2.hist(residuals, bins=20, alpha=0.7, color='blue', density=True)
                    ax2.set_xlabel('残差')
                    ax2.set_ylabel('频率')
                    ax2.set_title('残差分布')
                    
                    # 在直方图上添加KDE曲线
                    from scipy import stats
                    kde_x = np.linspace(min(residuals), max(residuals), 100)
                    kde = stats.gaussian_kde(residuals)
                    ax2.plot(kde_x, kde(kde_x), 'r-')
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                    
                    # 计算残差统计量
                    st.write("**残差统计分析**")
                    
                    residual_stats = {
                        "均值": np.mean(residuals),
                        "中位数": np.median(residuals),
                        "标准差": np.std(residuals),
                        "最小值": np.min(residuals),
                        "最大值": np.max(residuals),
                        "Q1 (25% 分位数)": np.percentile(residuals, 25),
                        "Q3 (75% 分位数)": np.percentile(residuals, 75)
                    }
                    
                    st.dataframe(pd.DataFrame([residual_stats]))
                    
                    # 检查残差是否为正态分布
                    k2, p = stats.normaltest(residuals)
                    st.write(f"正态性检验 p值: {p:.4f}")
                    if p < 0.05:
                        st.write("结论: 残差不符合正态分布 (p < 0.05)")
                    else:
                        st.write("结论: 残差符合正态分布 (p >= 0.05)")
                    
                    # 添加残差分析数据导出功能
                    with st.expander("导出残差分析数据"):
                        resid_data = pd.DataFrame({
                            '真实值': result['y_true'],
                            '预测值': result['极值'],
                            '残差': residuals
                        })
                        
                        # 检查NaN值
                        nan_count = resid_data.isna().sum().sum()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            resid_filename = st.text_input(
                                "文件名", 
                                f"{model_for_residuals}_残差分析_{datetime.now().strftime('%Y%m%d')}", 
                                key="analysis_resid_filename"  # 修改key，添加前缀以区分
                            )
                        with col2:
                            resid_format = st.selectbox(
                                "导出格式", 
                                ["Excel", "Origin格式(txt)"],
                                key="analysis_resid_format"  # 修改key，添加前缀以区分
                            )
                        
                        if st.button("导出残差分析数据", key="export_resid_button"):
                            try:
                                os.makedirs("exported_data", exist_ok=True)
                                if resid_format == "Excel":
                                    filepath = f"exported_data/{resid_filename}.xlsx"
                                    resid_data.to_excel(filepath, index=False)
                                else:  # Origin格式
                                    filepath = f"exported_data/{resid_filename}.txt"
                                    resid_data.to_csv(filepath, sep='\t', index=False, encoding='utf-8-sig')
                                st.success(f"数据已导出至: {filepath}")
                            except Exception as e:
                                st.error(f"导出数据时出错: {str(e)}")
                        
                        # 显示包含NaN的行
                        if nan_count > 0:
                            with st.expander("查看包含NaN的数据"):
                                nan_rows = resid_data[resid_data.isna().any(axis=1)]
                                st.write(f"包含NaN的行数: {len(nan_rows)}")
                                st.dataframe(nan_rows)
        else:
            st.warning("请先训练模型。")
    
    with tab5:
        st.header(t("model_saving"))
        
        if 'models' in st.session_state and st.session_state.models:
            st.subheader("模型保存")
            
            # 选择要保存的模型
            model_to_save = st.selectbox("选择要保存的模型", list(st.session_state.models.keys()), key="model_save_select")
            
            if model_to_save:
                # 模型保存选项
                col1, col2 = st.columns(2)
                
                with col1:
                    save_dir = st.text_input("保存目录", "models", key="model_save_dir")
                
                with col2:
                    model_filename = st.text_input("文件名", f"{model_to_save.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.pkl", key="model_filename")
                
                # 确保目录存在
                os.makedirs(save_dir, exist_ok=True)
                
                # 完整的保存路径
                save_path = os.path.join(save_dir, model_filename)
                
                # 模型保存按钮
                if st.button("保存模型", key="save_model_button"):
                    try:
                        # 获取模型
                        model = st.session_state.models[model_to_save]
                        
                        # 保存模型
                        joblib.dump(model, save_path)
                        st.success(f"模型已成功保存到: {save_path}")
                        
                        # 提供下载按钮
                        with open(save_path, "rb") as f:
                            st.download_button(
                                label="下载模型文件",
                                data=f,
                                file_name=model_filename,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"保存模型时出错: {str(e)}")
            
            # 在两列布局下方继续显示其他内容
            st.subheader("导出模型评估结果")
            # 创建评估结果表格
            evaluation_data = []
            for model_name, result in st.session_state.results.items():
                if 'error' not in result:  # 只显示成功训练的模型
                    evaluation_data.append({
                        '模型名称': model_name,
                        'MSE': result['mse'],
                        'R²': result['r2'],
                        'MAE': result['mae'],
                        '训练时间(秒)': result['training_time']
                    })
            
            if evaluation_data:
                eval_df = pd.DataFrame(evaluation_data)
                st.dataframe(eval_df)
                
                # 导出选项
                export_format = st.selectbox("导出格式", ["CSV", "Excel", "JSON"], key="results_export_format")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    export_dir = st.text_input("保存目录", "results", key="results_export_dir")
                
                with col2:
                    export_filename = st.text_input("文件名", f"model_evaluation_{datetime.now().strftime('%Y%m%d')}", key="results_filename")
                
                # 确保目录存在
                os.makedirs(export_dir, exist_ok=True)
                
                if st.button("导出结果", key="export_results_button"):
                    try:
                        # 根据选择的格式导出
                        if export_format == "CSV":
                            file_path = os.path.join(export_dir, f"{export_filename}.csv")
                            eval_df.to_csv(file_path, index=False)
                        elif export_format == "Excel":
                            file_path = os.path.join(export_dir, f"{export_filename}.xlsx")
                            eval_df.to_excel(file_path, index=False)
                        elif export_format == "JSON":
                            file_path = os.path.join(export_dir, f"{export_filename}.json")
                            eval_df.to_json(file_path, orient="records")
                        
                        st.success(f"评估结果已成功导出到: {file_path}")
                        
                        # 提供下载按钮
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="下载结果文件",
                                data=f,
                                file_name=os.path.basename(file_path),
                                mime="text/csv" if export_format == "CSV" else 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format == "Excel" else
                                     "application/json"
                            )
                    except Exception as e:
                        st.error(f"导出结果时出错: {str(e)}")
            
            st.subheader("一键导出所有图表数据")
            # 创建导出目录设置
            export_all_dir = st.text_input("导出目录", "all_charts_data", key="export_all_dir")
            
            if st.button("一键导出所有图表数据", key="export_all_charts"):
                try:
                    # 生成时间戳
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 定义目录结构
                    excel_dir = os.path.join(export_all_dir, "excel_data")
                    charts_dir = os.path.join(export_all_dir, "charts")
                    
                    # 创建所有必需的目录
                    for dir_path in [export_all_dir, excel_dir, charts_dir]:
                        os.makedirs(dir_path, exist_ok=True)
                        st.info(f"创建目录: {dir_path}")
                    
                    # 创建Excel写入器
                    excel_path = os.path.join(excel_dir, f"all_charts_data_{timestamp}.xlsx")
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        # 1. 模型性能比较数据
                        if 'results' in st.session_state and st.session_state.results:
                            performance_data = []
                            for model_name, result in st.session_state.results.items():
                                if 'error' not in result:
                                    performance_data.append({
                                        '模型名称': model_name,
                                        'MSE': result['mse'],
                                        'R²': result['r2'],
                                        'MAE': result['mae'],
                                        '训练时间(秒)': result['training_time']
                                    })
                            if performance_data:
                                perf_df = pd.DataFrame(performance_data)
                                perf_df.to_excel(writer, sheet_name='模型性能比较', index=False)
                                
                                # 保存性能比较图表
                                fig_perf, ax = plt.subplots(figsize=(12, 6))
                                perf_df.plot(kind='bar', x='模型名称', y=['MSE', 'R²', 'MAE'], ax=ax)
                                plt.title('模型性能对比', fontsize=12, fontproperties='SimHei')
                                plt.xlabel('模型', fontsize=10, fontproperties='SimHei')
                                plt.ylabel('指标值', fontsize=10, fontproperties='SimHei')
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                fig_perf.savefig(os.path.join(charts_dir, '模型性能比较.png'))
                                plt.close(fig_perf)
                        
                        # 2. 导出每个模型的预测结果和残差分析
                        for model_name, result in st.session_state.results.items():
                            if 'error' not in result:
                                # 预测结果数据
                                pred_df = pd.DataFrame({
                                    '真实值': result['y_true'],
                                    '预测值': result['极值'],
                                    '残差': result['y_true'] - result['极值']
                                })
                                pred_df.to_excel(writer, sheet_name=clean_sheet_name(f'{model_name}_预测结果'), index=False)
                                
                                # 预测结果散点图
                                fig_pred, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(result['y_true'], result['极值'], alpha=0.6)
                                ax.plot([min(result['y_true']), max(result['y_true'])],
                                       [min(result['y_true']), max(result['y_true'])],
                                       'r--', lw=2)
                                ax.set_xlabel('真实值', fontproperties='SimHei')
                                ax.set_ylabel('预测值', fontproperties='SimHei')
                                ax.set_title(f'{model_name}预测结果', fontproperties='SimHei')
                                plt.tight_layout()
                                fig_pred.savefig(os.path.join(charts_dir, f'{model_name}_预测结果.png'))
                                plt.close(fig_pred)
                                
                                # 残差分析图
                                fig_resid, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                
                                # 残差散点图
                                ax1.scatter(result['极值'], pred_df['残差'])
                                ax1.axhline(y=0, color='r', linestyle='--')
                                ax1.set_xlabel('预测值', fontproperties='SimHei')
                                ax1.set_ylabel('残差', fontproperties='SimHei')
                                ax1.set_title('残差 vs 预测值', fontproperties='SimHei')
                                
                                # 残差直方图
                                sns.histplot(pred_df['残差'], kde=True, ax=ax2)
                                ax2.set_xlabel('残差', fontproperties='SimHei')
                                ax2.set_ylabel('频率', fontproperties='SimHei')
                                ax2.set_title('残差分布', fontproperties='SimHei')
                                
                                plt.tight_layout()
                                fig_resid.savefig(os.path.join(charts_dir, f'{model_name}_残差分析.png'))
                                plt.close(fig_resid)
                        
                        # 3. 特征重要性分析
                        for model_name, model in st.session_state.models.items():
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    '特征': st.session_state.feature_cols,
                                    '重要性': model.feature_importances_
                                }).sort_values('重要性', ascending=False)
                                
                                importance_df.to_excel(writer, sheet_name=clean_sheet_name(f'{model_name}_特征重要性'), index=False)
                                
                                # 特征重要性图
                                fig_imp, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(data=importance_df, x='特征', y='重要性')
                                plt.xticks(rotation=45, ha='right')
                                plt.title(f'{model_name}特征重要性', fontproperties='SimHei')
                                plt.xlabel('特征', fontproperties='SimHei')
                                plt.ylabel('重要性', fontproperties='SimHei')
                                plt.tight_layout()
                                fig_imp.savefig(os.path.join(charts_dir, f'{model_name}_特征重要性.png'))
                                plt.close(fig_imp)
                        
                        # 4. 相关性分析
                        corr_matrix = pd.concat([st.session_state.X_train, st.session_state.y_train], axis=1).corr()
                        corr_matrix.to_excel(writer, sheet_name='相关性矩阵')
                        
                        # 相关性热图
                        fig_corr, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                        plt.title('特征相关性热图', fontproperties='SimHei')
                        plt.tight_layout()
                        fig_corr.savefig(os.path.join(charts_dir, '相关性热图.png'))
                        plt.close(fig_corr)
                        
                        # 5. 特征分布
                        for feature in st.session_state.feature_cols:
                            fig_dist, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(st.session_state.X_train[feature], kde=True, ax=ax)
                            ax.set_xlabel(feature, fontproperties='SimHei')
                            ax.set_ylabel('频率', fontproperties='SimHei')
                            ax.set_title(f'{feature}分布', fontproperties='SimHei')
                            plt.tight_layout()
                            safe_feature_name = clean_filename(feature)
                            fig_dist.savefig(os.path.join(charts_dir, f'{safe_feature_name}_分布.png'))
                            plt.close(fig_dist)
                        
                        # 6. 目标变量分布
                        fig_target, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(st.session_state.y_train, kde=True, ax=ax)
                        ax.set_xlabel(st.session_state.target_col, fontproperties='SimHei')
                        ax.set_ylabel('频率', fontproperties='SimHei')
                        ax.set_title('目标变量分布', fontproperties='SimHei')
                        plt.tight_layout()
                        fig_target.savefig(os.path.join(charts_dir, '目标变量分布.png'))
                        plt.close(fig_target)
                    
                    # 创建导出说明文档
                    readme_path = os.path.join(export_all_dir, f"README_{timestamp}.txt")
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write("图表数据导出说明\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("文件结构说明:\n")
                        f.write("1. Excel数据文件夹 (excel_data):\n")
                        f.write(f"   - all_charts_data_{timestamp}.xlsx 包含:\n")
                        f.write("     * 模型性能比较数据\n")
                        f.write("     * 各模型预测结果\n")
                        f.write("     * 特征重要性数据\n")
                        f.write("     * 相关性矩阵\n")
                        f.write("     * 特征分布数据\n\n")
                        f.write("2. 图表文件夹 (charts):\n")
                        f.write("   - 模型性能比较图\n")
                        f.write("   - 各模型预测结果图\n")
                        f.write("   - 残差分析图\n")
                        f.write("   - 特征重要性图\n")
                        f.write("   - 相关性热图\n")
                        f.write("   - 特征分布图\n")
                        f.write("   - 目标变量分布图\n\n")
                        f.write("图片格式: PNG\n")
                        f.write("分辨率: 300 DPI\n")
                    
                    st.success(f"""所有图表数据已成功导出！
                    \n- Excel文件保存在: {excel_path}
                    \n- 图表文件保存在: {charts_dir}
                    \n- 导出说明文件: {readme_path}""")
                    
                    # 提供下载按钮
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="下载Excel数据文件",
                            data=f,
                            file_name=f"all_charts_data_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                except Exception as e:
                    st.error(f"导出过程中出错: {str(e)}")
                    st.error("详细错误信息:")
                    st.exception(e)
            else:
                st.warning("请先训练模型。")

    # 添加新的预测标签页
    with tab6:
        st.header("模型预测")
        
        # 1. 选择模型来源
        st.subheader("1. 选择模型")
        uploaded_model = st.file_uploader("上传模型文件", type=['pkl', 'joblib'])
        if uploaded_model is not None:
            try:
                active_model = joblib.load(uploaded_model)
                st.success("模型加载成功！")
                
                # 2. 选择预测数据来源
                st.subheader("2. 选择预测数据来源")
                data_source = st.radio(
                    "选择数据来源",
                    ["上传新数据", "使用训练集", "使用测试集"],
                    key="data_source"
                )

                if data_source == "上传新数据":
                    uploaded_pred_file = st.file_uploader("上传预测数据", type=['csv', 'xlsx', 'xls'])
                    if uploaded_pred_file is not None:
                        try:
                            # 读取数据
                            file_extension = uploaded_pred_file.name.split('.')[-1].lower()
                            if file_extension == 'csv':
                                pred_data = pd.read_csv(uploaded_pred_file)
                            else:
                                pred_data = pd.read_excel(uploaded_pred_file)
                            st.success("数据加载成功！")
                            st.session_state['original_pred_data'] = pred_data.copy()
                        except Exception as e:
                            st.error(f"读取数据时出错：{str(e)}")
                            pred_data = None
                    else:
                        st.warning("请上传预测数据！")
                        pred_data = None
                elif data_source == "使用训练集":
                    if 'X_train' in st.session_state:
                        pred_data = st.session_state['X_train'].copy()
                        st.session_state['original_pred_data'] = pred_data.copy()
                        st.success("已加载训练集数据")
                    else:
                        st.error("训练集数据不可用！")
                        pred_data = None
                else:  # 使用测试集
                    if 'X_test' in st.session_state:
                        pred_data = st.session_state['X_test'].copy()
                        st.session_state['original_pred_data'] = pred_data.copy()
                        st.success("已加载测试集数据")
                    else:
                        st.error("测试集数据不可用！")
                        pred_data = None

                # 3. 数据预处理
                if pred_data is not None:
                    st.subheader("3. 数据预处理")
                    with st.expander("数据预处理选项", expanded=True):
                        st.write("请选择需要的预处理步骤：")
                        
                        # 处理缺失值
                        handle_missing = st.checkbox("处理缺失值", value=True)
                        if handle_missing:
                            missing_method = st.radio(
                                "缺失值处理方法",
                                ["删除含缺失值的行", "均值填充", "中位数填充", "0值填充"],
                                horizontal=True
                            )
                        
                        # 处理异常值
                        handle_outliers = st.checkbox("处理异常值")
                        if handle_outliers:
                            outlier_method = st.radio(
                                "异常值处理方法",
                                ["IQR方法", "Z-score方法"],
                                horizontal=True
                            )
                            if outlier_method == "Z-score方法":
                                z_threshold = st.slider("Z-score阈值", 2.0, 5.0, 3.0, 0.1)
                        
                        # 数据标准化/归一化
                        handle_scaling = st.checkbox("数据标准化/归一化")
                        if handle_scaling:
                            scaling_method = st.radio(
                                "标准化/归一化方法",
                                ["标准化(StandardScaler)", "最小最大归一化(MinMaxScaler)", "稳健归一化(RobustScaler)"],
                                horizontal=True
                            )
                        
                        # 应用预处理按钮
                        if st.button("应用预处理", key="apply_preprocessing"):
                            processed_data = pred_data.copy()
                            
                            # 处理缺失值
                            if handle_missing:
                                missing_count_before = processed_data.isna().sum().sum()
                                if missing_method == "删除含缺失值的行":
                                    processed_data = processed_data.dropna()
                                elif missing_method == "均值填充":
                                    processed_data = processed_data.fillna(processed_data.mean())
                                elif missing_method == "中位数填充":
                                    processed_data = processed_data.fillna(processed_data.median())
                                else:  # 0值填充
                                    processed_data = processed_data.fillna(0)
                                missing_count_after = processed_data.isna().sum().sum()
                                st.write(f"缺失值处理：从 {missing_count_before} 个减少到 {missing_count_after} 个")
                            
                            # 处理异常值
                            if handle_outliers:
                                outlier_count = 0
                                if outlier_method == "IQR方法":
                                    for column in processed_data.select_dtypes(include=[np.number]).columns:
                                        Q1 = processed_data[column].quantile(0.25)
                                        Q3 = processed_data[column].quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - 1.5 * IQR
                                        upper_bound = Q3 + 1.5 * IQR
                                        outlier_mask = (processed_data[column] < lower_bound) | (processed_data[column] > upper_bound)
                                        outlier_count += outlier_mask.sum()
                                        processed_data.loc[outlier_mask, column] = np.nan
                                else:  # Z-score方法
                                    for column in processed_data.select_dtypes(include=[np.number]).columns:
                                        z_scores = np.abs(stats.zscore(processed_data[column], nan_policy='omit'))
                                        outlier_mask = z_scores > z_threshold
                                        outlier_count += outlier_mask.sum()
                                        processed_data.loc[outlier_mask, column] = np.nan
                                
                                st.write(f"检测到 {outlier_count} 个异常值并已处理")
                                
                                # 处理异常值后产生的新缺失值
                                processed_data = processed_data.fillna(processed_data.median())
                            
                            # 数据标准化/归一化
                            if handle_scaling:
                                if scaling_method == "标准化(StandardScaler)":
                                    scaler = StandardScaler()
                                elif scaling_method == "最小最大归一化(MinMaxScaler)":
                                    scaler = MinMaxScaler()
                                else:  # 稳健归一化
                                    scaler = RobustScaler()
                                
                                # 对数值列进行缩放
                                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                                processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
                                st.write(f"已完成{scaling_method}处理")
                            
                            # 保存处理后的数据
                            st.session_state['processed_pred_data'] = processed_data
                            
                            # 显示处理后的数据统计信息
                            st.write("处理后的数据统计信息：")
                            st.write(processed_data.describe())
                            
                            # 显示处理前后对比
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("原始数据前5行：")
                                st.write(pred_data.head())
                            with col2:
                                st.write("处理后数据前5行：")
                                st.write(processed_data.head())
                            
                            st.success("数据预处理完成！")

                    # 4. 开始预测
                    st.subheader("4. 开始预测")
                    if st.button("开始预测", key="start_prediction"):
                        if 'processed_pred_data' not in st.session_state:
                            st.error("请先完成数据预处理！")
                        else:
                            try:
                                X_pred = st.session_state['processed_pred_data']
                                
                                # 显示特征列信息
                                st.write("当前数据的特征列：", list(X_pred.columns))
                                st.write("特征数量：", len(X_pred.columns))
                                
                                # 检查特征列是否匹配
                                if hasattr(active_model, 'n_features_in_'):
                                    expected_features = active_model.n_features_in_
                                    st.write("模型期望的特征数量：", expected_features)
                                    
                                    if len(X_pred.columns) != expected_features:
                                        st.error(f"特征数量不匹配！模型期望 {expected_features} 个特征，但数据包含 {len(X_pred.columns)} 个特征。")
                                        
                                        # 如果在session_state中存储了训练时的特征列名
                                        if 'feature_cols' in st.session_state:
                                            st.write("模型训练时使用的特征：", st.session_state.feature_cols)
                                            
                                            # 只选择训练时使用的特征列
                                            missing_features = set(st.session_state.feature_cols) - set(X_pred.columns)
                                            extra_features = set(X_pred.columns) - set(st.session_state.feature_cols)
                                            
                                            if missing_features:
                                                st.error(f"缺少以下特征：{missing_features}")
                                            if extra_features:
                                                st.warning(f"数据包含额外的特征：{extra_features}")
                                                st.info("将只使用训练时的特征进行预测")
                                            
                                            # 确保数据包含所有需要的特征
                                            if not missing_features:
                                                X_pred = X_pred[st.session_state.feature_cols]
                                                predictions = active_model.predict(X_pred)
                                                
                                                # 创建结果DataFrame
                                                results_df = pd.DataFrame({
                                                    '预测值': predictions
                                                })
                                                
                                                # 如果有真实值，添加到结果中
                                                if data_source in ["使用训练集", "使用测试集"]:
                                                    y_true = st.session_state['y_train' if data_source == "使用训练集" else 'y_test']
                                                    results_df['真实值'] = y_true
                                                    results_df['误差'] = results_df['真实值'] - results_df['预测值']
                                                
                                                # 保存结果
                                                st.session_state['prediction_results'] = results_df
                                                
                                                # 显示结果
                                                st.success("预测完成！")
                                                st.write("预测结果：")
                                                st.dataframe(results_df)
                                            else:
                                                st.error("无法进行预测，因为缺少必要的特征")
                                        else:
                                            st.error("无法确定模型训练时使用的特征列，请确保使用相同的特征集进行预测")
                                    else:
                                        # 特征数量匹配，直接进行预测
                                        predictions = active_model.predict(X_pred)
                                        
                                        # 创建结果DataFrame
                                        results_df = pd.DataFrame({
                                            '预测值': predictions
                                        })
                                        
                                        # 如果有真实值，添加到结果中
                                        if data_source in ["使用训练集", "使用测试集"]:
                                            y_true = st.session_state['y_train' if data_source == "使用训练集" else 'y_test']
                                            results_df['真实值'] = y_true
                                            results_df['误差'] = results_df['真实值'] - results_df['预测值']
                                        
                                        # 保存结果
                                        st.session_state['prediction_results'] = results_df
                                        
                                        # 显示结果
                                        st.success("预测完成！")
                                        st.write("预测结果：")
                                        st.dataframe(results_df)
                                else:
                                    st.error("无法确定模型的特征数量，请确保使用正确的模型文件")
                                    
                            except Exception as e:
                                st.error(f"预测过程中出错：{str(e)}")
                                st.exception(e)
            except Exception as e:
                st.error(f"加载模型时出错：{str(e)}")
        else:
            st.warning("请先上传模型文件！")

# 预测结果可视化分析
if 'prediction_results' in st.session_state and not st.session_state['prediction_results'].empty:
    st.subheader("预测结果可视化分析")
    results_df = st.session_state['prediction_results']
    
    # 添加导入真实值功能
    with st.expander("导入真实值数据", expanded=True):
        st.write("如果有真实值数据，可以导入进行对比分析")
        true_values_file = st.file_uploader("上传真实值数据文件", type=['csv', 'xlsx', 'xls'], key="true_values_uploader")
        
        if true_values_file is not None:
            try:
                # 读取真实值数据
                file_extension = true_values_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    true_values_df = pd.read_csv(true_values_file)
                else:
                    true_values_df = pd.read_excel(true_values_file)
                
                # 显示原始数据信息
                st.write("原始数据信息：")
                st.write(f"- 总行数：{len(true_values_df)}")
                st.write(f"- 包含缺失值的行数：{true_values_df.isna().any(axis=1).sum()}")
                
                # 添加删除缺失值选项
                handle_missing = st.checkbox("删除包含缺失值的行", value=True)
                if handle_missing:
                    # 保存原始数据的副本
                    original_df = true_values_df.copy()
                    # 删除包含缺失值的行
                    true_values_df = true_values_df.dropna()
                    # 显示处理后的信息
                    st.write("处理后的数据信息：")
                    st.write(f"- 剩余行数：{len(true_values_df)}")
                    st.write(f"- 删除的行数：{len(original_df) - len(true_values_df)}")
                    
                    # 显示数据预览
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("原始数据预览：")
                        st.dataframe(original_df.head())
                    with col2:
                        st.write("处理后数据预览：")
                        st.dataframe(true_values_df.head())
                else:
                    # 显示原始数据预览
                    st.write("数据预览：")
                    st.dataframe(true_values_df.head())
                
                # 选择真实值列
                true_value_column = st.selectbox(
                    "选择真实值列",
                    true_values_df.columns,
                    key="true_value_column"
                )
                
                if st.button("确认添加真实值", key="confirm_true_values"):
                    # 检查数据长度是否匹配
                    if len(true_values_df) != len(results_df):
                        st.error(f"""
                        数据长度不匹配！
                        - 真实值数据长度：{len(true_values_df)}
                        - 预测结果长度：{len(results_df)}
                        请确保数据长度相同。
                        """)
                    else:
                        # 检查选择的列是否包含缺失值
                        if true_values_df[true_value_column].isna().any():
                            st.warning(f"选择的列'{true_value_column}'包含缺失值，这可能会影响分析结果")
                        
                        # 更新results_df，添加真实值和误差
                        results_df['真实值'] = true_values_df[true_value_column]
                        results_df['误差'] = results_df['真实值'] - results_df['预测值']
                        st.session_state['prediction_results'] = results_df
                        st.success(f"""
                        成功添加真实值数据！
                        - 使用列：{true_value_column}
                        - 数据行数：{len(results_df)}
                        """)
                        
                        # 显示更新后的数据预览
                        st.write("更新后的预测结果预览：")
                        st.dataframe(results_df.head())
                    
            except Exception as error:  # 修改这里，使用不同的变量名
                st.error(f"读取真实值数据时出错：{str(error)}")
                st.exception(error)  # 使用正确的变量名
    
    # 创建多个标签页进行不同的可视化分析
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "预测vs真实值分析", 
        "误差分析", 
        "预测值分布",
        "统计指标"
    ])
    
    # 标签页1：预测vs真实值分析
    with viz_tab1:
        if '真实值' in results_df.columns:
            st.write("### 预测值 vs 真实值")
            
            # 散点图
            fig_scatter = px.scatter(
                results_df, 
                x='真实值', 
                y='预测值',
                title="预测值 vs 真实值散点图"
            )
            
            # 添加对角线 (y=x)
            x_range = [results_df['真实值'].min(), results_df['真实值'].max()]
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_range,
                    y=x_range,
                    mode='lines',
                    name='理想预测线',
                    line=dict(color='red', dash='dash')
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 添加导出功能
            if st.button("导出散点图数据", key="export_scatter"):
                scatter_data = pd.DataFrame({
                    '真实值': results_df['真实值'],
                    '预测值': results_df['预测值']
                })
                st.download_button(
                    label="下载散点图数据(Excel)",
                    data=convert_df_to_excel(scatter_data),
                    file_name=f"预测vs真实值散点图数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("请先导入真实值数据进行对比分析")
    
    # 标签页2：误差分析
    with viz_tab2:
        if '误差' in results_df.columns:
            st.write("### 误差分析")
            
            # 误差直方图
            fig_hist = px.histogram(
                results_df, 
                x='误差',
                nbins=30,
                title="预测误差分布直方图"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 误差箱线图
            fig_box = px.box(
                results_df,
                y='误差',
                title="预测误差箱线图"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 残差图（预测值vs误差）
            fig_resid = px.scatter(
                results_df,
                x='预测值',
                y='误差',
                title="残差图（预测值 vs 误差）"
            )
            # 添加y=0线
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_resid, use_container_width=True)
            
            # 添加导出功能
            if st.button("导出误差分析数据", key="export_error"):
                error_data = pd.DataFrame({
                    '预测值': results_df['预测值'],
                    '误差': results_df['误差']
                })
                st.download_button(
                    label="下载误差分析数据(Excel)",
                    data=convert_df_to_excel(error_data),
                    file_name=f"误差分析数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("没有误差数据，无法进行误差分析")
    
    # 标签页3：预测值分布
    with viz_tab3:
        st.write("### 预测值分布")
        
        # 预测值直方图
        fig_pred_hist = px.histogram(
            results_df,
            x='预测值',
            nbins=30,
            title="预测值分布直方图"
        )
        st.plotly_chart(fig_pred_hist, use_container_width=True)
        
        # 预测值箱线图
        fig_pred_box = px.box(
            results_df,
            y='预测值',
            title="预测值箱线图"
        )
        st.plotly_chart(fig_pred_box, use_container_width=True)
        
        # 添加导出功能
        if st.button("导出预测值分布数据", key="export_dist"):
            dist_data = pd.DataFrame({
                '预测值': results_df['预测值']
            })
            st.download_button(
                label="下载预测值分布数据(Excel)",
                data=convert_df_to_excel(dist_data),
                file_name=f"预测值分布数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # 标签页4：统计指标
    with viz_tab4:
        st.write("### 预测性能统计指标")
        
        if '真实值' in results_df.columns:
            # 计算各种统计指标
            metrics = calculate_metrics(
                results_df['真实值'].values,
                results_df['预测值'].values
            )
            
            # 创建指标展示的列
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("均方误差 (MSE)", f"{metrics['mse']:.4f}")
                st.metric("均方根误差 (RMSE)", f"{metrics['rmse']:.4f}")
                st.metric("平均绝对误差 (MAE)", f"{metrics['mae']:.4f}")
                st.metric("最大绝对误差", f"{metrics['max_error']:.4f}")
                st.metric("最小绝对误差", f"{metrics['min_error']:.4f}")
            
            with col2:
                st.metric("决定系数 (R²)", f"{metrics['r2']:.4f}")
                st.metric("调整后的R² (Adjusted R²)", f"{metrics['adj_r2']:.4f}")
                st.metric("相关系数", f"{metrics['correlation']:.4f}")
                st.metric("平均绝对百分比误差 (MAPE)", f"{metrics['mape']:.2f}%")
                st.metric("标准误差", f"{metrics['std_error']:.4f}")
            
            # 添加导出功能
            if st.button("导出统计指标", key="export_metrics"):
                metrics_df = pd.DataFrame([metrics])
                st.download_button(
                    label="下载统计指标(Excel)",
                    data=convert_df_to_excel(metrics_df),
                    file_name=f"预测性能统计指标_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("没有真实值数据，无法计算统计指标")

else:
    st.warning("请先进行预测！")

# 辅助函数：计算统计指标
def calculate_metrics(y_true, y_pred):
    """
    计算预测性能的各种统计指标
    """
    metrics = {}
    
    try:
        # 基础指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 计算MAPE (平均绝对百分比误差)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
        
        # 计算调整后的R² (Adjusted R-squared)
        n = len(y_true)
        p = 1  # 预测变量的数量
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        # 计算相关系数
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 计算最大误差和最小误差
        errors = y_true - y_pred
        metrics['max_error'] = np.max(np.abs(errors))
        metrics['min_error'] = np.min(np.abs(errors))
        
        # 计算标准误差
        metrics['std_error'] = np.std(errors)
        
    except Exception as error:
        st.error(f"计算统计指标时出错：{str(error)}")
        metrics = {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'mape': np.nan,
            'adj_r2': np.nan,
            'correlation': np.nan,
            'max_error': np.nan,
            'min_error': np.nan,
            'std_error': np.nan
        }
    
    return metrics

# 辅助函数：将DataFrame转换为Excel文件
def convert_df_to_excel(df):
    """将DataFrame转换为Excel文件"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()