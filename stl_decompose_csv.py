import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decompose_df(df, period=12):
    """
    对DataFrame的每一列做STL分解，返回分解后的DataFrame。
    每个特征分解为三列：特征名_T, 特征名_S, 特征名_R
    """
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col].dropna()
        # STL要求长度大于2*period
        if len(series) < 2 * period:
            print(f"列 {col} 长度不足，跳过")
            continue
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        # 结果对齐原始索引
        trend = pd.Series(res.trend, index=series.index)
        seasonal = pd.Series(res.seasonal, index=series.index)
        resid = pd.Series(res.resid, index=series.index)
        result[f"{col}_T"] = trend
        result[f"{col}_S"] = seasonal
        result[f"{col}_R"] = resid
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对CSV每列做STL分解")
    parser.add_argument("--input_csv", default='/mnt/raid/zhengsen/pixel_5_176_year_2024.csv', help="输入CSV文件路径")
    parser.add_argument("--output_csv", default='/mnt/raid/zhengsen/pixel_5_176_year_2024_stl.csv', help="输出CSV文件路径")
    parser.add_argument("--period", type=int, default=12, help="季节性周期长度，默认12")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    result = stl_decompose_df(df, period=args.period)
    result.to_csv(args.output_csv, index=False)
    print(f"分解结果已保存到 {args.output_csv}") 