def truncate_patient(patient, max_len: int, background_tokens_per_patient, sep_token=2):
    total_length = len(patient["concept"])
    if total_length <= max_len:
        return patient

    # Determine how many items from the end we can keep
    tail_length = max_len - background_tokens_per_patient

    # If the boundary element is the SEP token, shift tail_length by 1
    if tail_length > 0 and patient["concept"][-tail_length] == sep_token:
        tail_length = max(tail_length - 1, 0)

    for key, val in patient.items():
        if isinstance(val, list):
            patient[key] = val[:background_tokens_per_patient] + val[-tail_length:]

    return patient
