def test_column_names(data):

    expected_columns = [
            "age",
            "workclass",
            "fnlgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "salary"
    ]

    these_columns = data.columns.values

    assert list(expected_columns) == list(these_columns)
