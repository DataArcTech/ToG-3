import json

input_path = "./data/multimodal_test_samples/documents.json"
output_path = "./data/multimodal_test_samples/samples.jsonl"


def convert_json_to_jsonl(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    # 首先根据item["metadata"]["source"]去重文章
    articles = {}
    for item in data:
        source = item["metadata"]["source"]
        if source not in articles:
            title = source
            text = item["metadata"]["header"] + "\n" + item["text"]
            articles[source] = [title, [text]]
        else:
            text = item["metadata"]["header"] + "\n" + item["text"]
            articles[source][1].append(text)

    new_data = []

    for source, texts in articles.items():

        new_item = {
            "_id": source,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "evidences": item.get("evidences", []),
            "context": texts
        }
        new_data.append(new_item)


    # Write JSONL data to output file
    with open(output_path, "w") as f:
        for item in new_data:
            f.write(json.dumps(item, indent=4) + "\n")


if __name__ == "__main__":
    # Convert JSON to JSONL
    print(f"Converting {input_path} to {output_path}...")
    convert_json_to_jsonl(input_path, output_path)
