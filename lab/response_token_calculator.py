

def token_calc(response):
    token_usage = {
        "input_tokens" : response.usage_metadata.get("input_tokens"),
        "output_tokens" : response.usage_metadata.get("output_tokens"),
        "total_tokens" : response.usage_metadata.get("total_tokens")
    }

    return token_usage