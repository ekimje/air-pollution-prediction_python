def result_append(model_name, model_result):
    result = []
    for data_name, m_result in model_result.items():
        result.append({
            "model":model_name,
            "Data":data_name,
            **m_result
        })
    return result
