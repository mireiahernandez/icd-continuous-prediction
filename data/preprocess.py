import pandas as pd
import os
import ast


class DataProcessor:
    """Prepare data for model.

    Some of the functions of this class are extracted from the HTDC (Ng et al, 2023):
    aggregate_hadm_id, add_category_information and multi_hot_encode.

    The contributions of this work is to add the temporal information.
    """

    def __init__(self, dataset_path, config):        
        self.notes_df = pd.read_csv(os.path.join(dataset_path, "NOTEEVENTS.csv"))
        if config["debug"]:
            self.notes_df = self.notes_df.sort_values(by='HADM_ID')[:3000]
        self.labels_df = pd.read_csv(
            os.path.join(dataset_path, "splits/caml_splits.csv")
        )
        self.config = config
        self.filter_discharge_summary()

    def aggregate_data(self):
        """Preprocess data and aggregate."""
        notes_agg_df = self.aggregate_hadm_id()
        notes_agg_df = self.add_category_information(notes_agg_df)
        notes_agg_df = self.add_temporal_information(notes_agg_df)
        notes_agg_df = self.add_multi_hot_encoding(notes_agg_df)
        return notes_agg_df

    def filter_discharge_summary(self):
        """Filter only DS if needed.
        Based on HTDC
        """
        if self.config["only_discharge_summary"]:
            self.notes_df = self.notes_df[self.notes_df.CATEGORY == "Discharge summary"]

    def aggregate_hadm_id(self):
        """Aggregate all notes of the same HADM_ID
        Based on HTDC
        """
        # Filter NA hadm_id
        self.notes_df = self.notes_df[self.notes_df.HADM_ID.isna() == False]
        self.notes_df["HADM_ID"] = self.notes_df["HADM_ID"].apply(int)

        # if time is missing -> assume 12:00:00
        ### MY CONTRIBUTION #############33
        self.notes_df.CHARTTIME = self.notes_df.CHARTTIME.fillna(
            self.notes_df.CHARTDATE + " 12:00:00"
        )
        self.notes_df["CHARTTIME"] = pd.to_datetime(self.notes_df.CHARTTIME)
        #################

        self.notes_df["is_discharge_summary"] = (
            self.notes_df.CATEGORY == "Discharge summary"
        )
        notes_agg_df = (
            self.notes_df.sort_values(
                by=["CHARTDATE", "CHARTTIME", "is_discharge_summary"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({"TEXT": list, "CHARTDATE": list, "CHARTTIME": list, "CATEGORY": list})
        ).reset_index()

        notes_agg_df = notes_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")

        notes_agg_df = notes_agg_df[notes_agg_df.SPLIT_50.isna() != True]
        return notes_agg_df

    def _timedelta_to_hours(self, timedelta):
        td = timedelta.components
        return td.days * 24 + td.hours + td.minutes / 60.0 + td.seconds / 3600.0

    def add_temporal_information(self, notes_agg_df):
        """Add time information."""
        # Add temporal information
        notes_agg_df["ADMISSION_DATETIME"] = notes_agg_df["CHARTTIME"].apply(
            lambda s: s[0]
        )
        notes_agg_df["TIME_ELAPSED"] = notes_agg_df["CHARTTIME"].apply(
            lambda s: [s[i] - s[0] for i in range(len(s))]
        )
        notes_agg_df["PERCENT_ELAPSED"] = notes_agg_df["CHARTTIME"].apply(
            lambda s: [
                (s[i] - s[0]) / (s[-1] - s[0]) if s[-1] - s[0] > pd.Timedelta(0) else 1
                for i in range(len(s) - 1)
            ]
        )
        notes_agg_df["HOURS_ELAPSED"] = notes_agg_df["TIME_ELAPSED"].apply(
            lambda s: [self._timedelta_to_hours(td) for td in s]
        )
        return notes_agg_df

    def _func(self, x):
        if x == 0:
            return 2
        else:
            return x

    def _get_reverse_seqid_by_category(self, category_ids):
        # This creates the CATEGORY_REVERSE_SEQID field for use in note selection later
        # For each category, the last note is assigned to index 0, the second last note is assigned index 1, and so on
        category_ids = pd.Series(category_ids)
        category_ranks = category_ids.groupby(category_ids).cumcount(ascending=False)
        return list(category_ranks)

    def add_category_information(self, notes_agg_df):
        # Create Category IDs
        categories = list(
            notes_agg_df["CATEGORY"]
            .apply(lambda x: pd.Series(x))
            .stack()
            .value_counts()
            .index
        )
        categories_mapping = {categories[i]: i for i in range(len(categories))}
        print(categories_mapping)

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY"].apply(
            lambda x: [categories_mapping[c] for c in x]
        )

        # The "Nursing/Other" category is present in the train set but not the dev/test sets
        # We group them together with notes in the "Nursing" category as described in our paper

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY_INDEX"].apply(
            lambda x: [self._func(y) for y in x]
        )

        notes_agg_df["CATEGORY_REVERSE_SEQID"] = notes_agg_df["CATEGORY_INDEX"].apply(
            self._get_reverse_seqid_by_category
        )
        return notes_agg_df

    def _multi_hot_encode(self, codes, code_counts):
        """Return a multi hot encoded vector.

        The resulting multi-hot encoded vector contains ALL labels.
        The top 50 labels can then be filtered using [:50]

        Args:
            codes (list): sample labels
            code_counts (pd.series): series mapping code to frequency

        Return:
            multi_hot (list): list of 0s and 1s
        """
        res = []

        ref = code_counts.index.tolist()

        for c in ref:
            if c in codes:
                res.append(1.0)
            else:
                res.append(0.0)

        return res

    def add_multi_hot_encoding(self, notes_agg_df):
        # liter eval: evaluate the string into a list
        notes_agg_df["absolute_code"] = notes_agg_df["absolute_code"].apply(
            lambda x: ast.literal_eval(x)
        )
        code_counts = (
            notes_agg_df["absolute_code"]
            .apply(lambda x: pd.Series(x))
            .stack()
            .value_counts()
        )

        notes_agg_df["ICD9_CODE_BINARY"] = notes_agg_df["absolute_code"].apply(
            lambda x: self._multi_hot_encode(x, code_counts)
        )

        # We focus on the MIMIC-III-50 splits
        notes_agg_df["SPLIT"] = notes_agg_df["SPLIT_50"]

        notes_agg_df = notes_agg_df[notes_agg_df.SPLIT.isna() != True]

        return notes_agg_df
