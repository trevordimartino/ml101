import torch


def generate_filler_class_names(n_classes):
    return {n: f'class{n}' for n in range(n_classes)}


def get_confusion_matrix(model, dataloader):
    # Note: Model class must have an `n_classes` attribute
    confusion_matrix = torch.zeros(model.n_classes, model.n_classes)
    with torch.no_grad():
        for val_batch, (inputs, targets) in enumerate(dataloader, 0):
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            for guess, target in zip(predicted.tolist(), targets.tolist()):
                confusion_matrix[guess, target] += 1

    return confusion_matrix


def print_accuracy(confusion_matrix, true_class_dim=1):
    total = torch.sum(confusion_matrix)
    total_correct = torch.sum(torch.diag(confusion_matrix))
    print(f'Accuracy (overall): {100 * total_correct.item() / total.item():.2f}%')


def print_class_recalls(confusion_matrix, class_from_idx=None, guess_class_dim=0):
    class_correct = torch.diag(confusion_matrix)
    class_totals = torch.sum(confusion_matrix, guess_class_dim)
    class_recalls = class_correct / class_totals

    if not class_from_idx:
        class_from_idx = generate_filler_class_names(len(confusion_matrix))

    # TODO (TD): Make ljust on label names more robust
    for (_, label), pct in zip(class_from_idx.items(), class_recalls.tolist()):
        print(f'Recall for {label.ljust(3)}: {100 * pct:.2f}%')

    return class_recalls


def print_class_precisions(confusion_matrix, class_from_idx=None, true_class_dim=1):
    class_correct = torch.diag(confusion_matrix)
    guess_totals = torch.sum(confusion_matrix, true_class_dim)
    class_precisions = class_correct / guess_totals

    if not class_from_idx:
        class_from_idx = generate_filler_class_names(len(confusion_matrix))

    for (_, label), pct in zip(class_from_idx.items(), class_precisions.tolist()):
        print(f'Precision for {label.ljust(3)}: {100 * pct:.2f}%')

    return class_precisions


def print_f1_scores(confusion_matrix, class_from_idx=None, true_class_dim=1, guess_class_dim=0):
    class_recalls = print_class_recalls(
        confusion_matrix,
        class_from_idx=class_from_idx,
        guess_class_dim=guess_class_dim
    )
    class_precisions = print_class_precisions(
        confusion_matrix,
        class_from_idx=class_from_idx,
        true_class_dim=true_class_dim
    )

    class_f1s = 2 * (class_precisions * class_recalls) / (class_precisions + class_recalls)

    if not class_from_idx:
        class_from_idx = generate_filler_class_names(len(confusion_matrix))

    for (_, label), f1 in zip(class_from_idx.items(), class_f1s):
        print(f'F1 for {label.ljust(3)}: {f1:.6f}')
