def result_append(model_name, model_result):
    result = []
    for data_name, predict_result in model_result.items():
        result.append({
            "model":model_name,
            "Data":data_name,
            **predict_result
        })
    return result
