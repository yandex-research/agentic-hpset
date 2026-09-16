from .cat_pair_concat import cat_pair_concat_features
from .cat_target_std import cat_target_std_features
from .frequency_encoding import frequency_encoding_features
from .groupby_mean import groupby_mean_features
from .hash_high_card_cat import hash_high_card_cat_features
from .identity import identity_features
from .kmeans_cluster import kmeans_cluster_features
from .log1p_skew import log1p_skew_features
from .nan_indicator import nan_indicator_features
from .pairwise_difference import pairwise_difference_features
from .pairwise_product import pairwise_product_features
from .pca_components import pca_components_features
from .quantile_bin import quantile_bin_features
from .rank_percentile import rank_percentile_features
from .rare_category_collapse import rare_category_collapse_features
from .row_stats import row_stats_features
from .target_mean_encoding import target_mean_encoding_features

FEATURE_MAP = {
    0: identity_features,
    1: quantile_bin_features,
    2: pairwise_product_features,
    3: row_stats_features,
    4: nan_indicator_features,
    5: frequency_encoding_features,
    6: target_mean_encoding_features,
    7: pca_components_features,
    8: kmeans_cluster_features,
    9: cat_pair_concat_features,
    10: groupby_mean_features,
    11: log1p_skew_features,
    12: rare_category_collapse_features,
    13: pairwise_difference_features,
    14: hash_high_card_cat_features,
    15: rank_percentile_features,
    16: cat_target_std_features,
}

__all__ = [
    "FEATURE_MAP",
    "cat_pair_concat_features",
    "cat_target_std_features",
    "frequency_encoding_features",
    "groupby_mean_features",
    "hash_high_card_cat_features",
    "identity_features",
    "kmeans_cluster_features",
    "log1p_skew_features",
    "nan_indicator_features",
    "pairwise_difference_features",
    "pairwise_product_features",
    "pca_components_features",
    "quantile_bin_features",
    "rank_percentile_features",
    "rare_category_collapse_features",
    "row_stats_features",
    "target_mean_encoding_features",
]
