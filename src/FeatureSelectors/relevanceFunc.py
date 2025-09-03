"""
This module provides functions for calculating feature relevance using statistical methods.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def oneWayANOVA(df: DataFrame, features: list, label: str) -> DataFrame:
    """
    Calculates the F-value for each feature using one-way ANOVA.

    Args:
        df (DataFrame): The input DataFrame.
        features (list): A list of feature columns to calculate the F-value for.
        label (str): The name of the label column.

    Returns:
        DataFrame: A DataFrame containing the F-value for each feature.
    """

    # get the number of classes
    n_classes = df.select(F.countDistinct(label)).collect()[0][0]
    # get number of samples
    n_samples = df.count()

    # Calculations that does not need grouping
    non_grouping_expr = []
    for feature in features:
        non_grouping_expr.append(
            F.sum(F.col(feature) ** 2).alias(f"sum_of_square_{feature}")
        )
        non_grouping_expr.append(
            (F.sum(F.col(feature)) ** 2).alias(f"square_of_sum_{feature}")
        )
    df_non_group_expr = df.select(*non_grouping_expr)

    # Calculations that need grouping
    grouping_expr = []
    for feature in features:
        grouping_expr.append(F.sum(feature).alias(f"grouped_sum_{feature}"))
    df_group_expr = df.groupBy(label).agg(*grouping_expr, F.count("*").alias("count"))

    # calculate sst
    sst_expr = []
    for feature in features:
        sst_expr.append(
            (
                F.col(f"sum_of_square_{feature}")
                - F.col(f"square_of_sum_{feature}") / F.lit(n_samples)
            ).alias(f"sst_{feature}")
        )
    df_sst = df_non_group_expr.select(*sst_expr)

    # Calculate pre ssbn
    pre_ssbn_expr = []
    for feature in features:
        pre_ssbn_expr.append(
            (F.sum(F.col(f"grouped_sum_{feature}") ** 2 / F.col("count"))).alias(
                f"pre_ssbn_{feature}"
            )
        )
    df_pre_ssbn = df_group_expr.select(*pre_ssbn_expr)

    # calculate ssbn
    # join two tables
    df_pre_ssbn = df_pre_ssbn.withColumn("key", F.col(f"pre_ssbn_{features[0]}") * 0)
    df_non_group_expr = df_non_group_expr.withColumn(
        "key", F.col(f"square_of_sum_{features[0]}") * 0
    )
    df_join = df_pre_ssbn.join(df_non_group_expr, on="key")
    # calculation expressions
    ssbn_expr = []
    for feature in features:
        ssbn_expr.append(
            (
                F.col(f"pre_ssbn_{feature}")
                - F.col(f"square_of_sum_{feature}") / F.lit(n_samples)
            ).alias(f"ssbn_{feature}")
        )
    df_ssbn = df_join.select(*ssbn_expr, F.col("key"))

    # calculate sswn
    df_sst = df_sst.withColumn("key", F.col(f"sst_{features[0]}") * 0)
    df_join = df_ssbn.join(df_sst, on="key")
    # sswn calculation expression
    sswn_expr = []
    for feature in features:
        sswn_expr.append(
            (F.col(f"sst_{feature}") - F.col(f"ssbn_{feature}")).alias(
                f"sswn_{feature}"
            )
        )
    df_sswn = df_join.select(*sswn_expr, F.col("key"))

    # Calculate msb
    msb_expr = []
    for feature in features:
        msb_expr.append(
            (F.col(f"ssbn_{feature}") / F.lit(n_classes - 1)).alias(f"msb_{feature}")
        )
    df_msb = df_ssbn.select(*msb_expr, F.col("key"))

    # Calculate msw
    msw_expr = []
    for feature in features:
        msw_expr.append(
            (F.col(f"sswn_{feature}") / (F.lit(n_samples) - F.lit(n_classes))).alias(
                f"msw_{feature}"
            )
        )
    df_msw = df_sswn.select(*msw_expr, F.col("key"))

    # Final Fvalue
    df_join = df_msb.join(df_msw, on="key")
    fvalue_expr = []
    for feature in features:
        fvalue_expr.append(
            (F.col(f"msb_{feature}") / F.col(f"msw_{feature}")).alias(f"{feature}")
        )
    df_fvalue = df_join.select(*fvalue_expr)

    return df_fvalue
