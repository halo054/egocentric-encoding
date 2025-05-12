import torch
import pathlib

group_1_dir = "embeddings/exo/"
group_2_dir = "embeddings/ego-original/"

group_1_labels = [f.name.replace('.pt', '') for f in pathlib.Path(group_1_dir).iterdir() if f.is_file()]
group_1_features = [torch.load(f"{group_1_dir}/{f}.pt", weights_only=False)[0] for f in group_1_labels]

group_2_labels = group_1_labels
group_2_features = [torch.load(f"{group_2_dir}/{f}.pt", weights_only=False)[0] for f in group_2_labels]

group_1_features = torch.stack(group_1_features)
group_2_features = torch.stack(group_2_features)

cos_sim = torch.nn.functional.cosine_similarity(group_1_features, group_2_features)

print(f"Average cosine similarity score: {torch.mean(cos_sim)}")

# return (correct_1, correct_5, correct_10)
def retrieve(query, query_label, db, db_labels):
    similarity = torch.squeeze(torch.mm(torch.unsqueeze(query, dim=0), db.t()))
    _, indices = similarity.sort(dim=0, descending=True)

    correct_1 = False
    correct_5 = False
    correct_10 = False
    top10 = [db_labels[j] for j in indices[:10].cpu().numpy()]
    # get the R@k
    if query_label == top10[0]:
        correct_1 = True
    if query_label in top10[:5]:
        correct_5 = True
    if query_label in top10:
        correct_10 = True
    
    return (correct_1, correct_5, correct_10)

correct_1 = 0
correct_5 = 0
correct_10 = 0
for idx, label in enumerate(group_1_labels):
    ret = retrieve(group_1_features[idx], label, group_2_features, group_2_labels)
    correct_1 += 1 if ret[0] else 0
    correct_5 += 1 if ret[1] else 0
    correct_10 += 1 if ret[2] else 0

print(f"=====Retrieving {group_2_dir} from {group_1_dir}=====")
print(f"R@1: {correct_1 / len(group_1_labels)}")
print(f"R@5: {correct_5 / len(group_1_labels)}")
print(f"R@10: {correct_10 / len(group_1_labels)}")

correct_1 = 0
correct_5 = 0
correct_10 = 0
for idx, label in enumerate(group_2_labels):
    ret = retrieve(group_2_features[idx], label, group_1_features, group_1_labels)
    correct_1 += 1 if ret[0] else 0
    correct_5 += 1 if ret[1] else 0
    correct_10 += 1 if ret[2] else 0
print(f"=====Retrieving {group_1_dir} from {group_2_dir}=====")
print(f"R@1: {correct_1 / len(group_2_labels)}")
print(f"R@5: {correct_5 / len(group_2_labels)}")
print(f"R@10: {correct_10 / len(group_2_labels)}")
