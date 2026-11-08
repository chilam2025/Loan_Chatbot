import pandas as pd

from abc_credit.modeling.compare_pincode_signal import represent_pincode


def test_pincode_prefix_preserves_leading_zeros_and_handles_missing_values():
    applications = pd.DataFrame({
        "Pincode": [" 110001 ", "001234", "110", "ABC123", "", None],
    })

    result = represent_pincode(applications, "prefix_3")

    assert result["Pincode"].iloc[:2].tolist() == ["110", "001"]
    assert result["Pincode"].iloc[2:].isna().all()
    assert applications["Pincode"].iloc[0] == " 110001 "