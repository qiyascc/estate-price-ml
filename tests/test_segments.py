from estate_price_ml import features, segments


def test_split_sale(raw_df):
    frame = features.clean(raw_df)
    sale = segments.split_segment(frame, "satis")
    assert len(sale) > 0
    assert (sale["deal_type"] == "sale").all()


def test_owner_sale_segment(raw_df):
    frame = features.clean(raw_df)
    part = segments.split_segment(frame, "sahibkar_satis")
    assert (part["deal_type"] == "sale").all()
    assert (part["owner_type"] == "owner").all()


def test_agent_sale_segment(raw_df):
    frame = features.clean(raw_df)
    part = segments.split_segment(frame, "vasitechi_satis")
    assert (part["deal_type"] == "sale").all()
    assert (part["owner_type"] == "agent").all()


def test_all_segment_keys(raw_df):
    frame = features.clean(raw_df)
    result = segments.all_segments(frame)
    assert set(result.keys()) == {"satis", "kiraye", "sahibkar_satis", "vasitechi_satis"}
