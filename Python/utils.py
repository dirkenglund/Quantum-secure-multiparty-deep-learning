try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # Torch may not be available in minimal environments
    torch = None
    TORCH_AVAILABLE = False


def calculate_accuracy(loader, model, secure=False, printprogress=False):
    """Compute classification accuracy.

    Parameters
    ----------
    loader : iterable
        Iterable yielding ``(images, labels)`` pairs.
    model : object
        Model callable returning network outputs. It should expose
        ``eval`` and ``train`` methods.
    secure : bool, optional
        If ``True``, the model returns ``(outputs, PDmeans, PDrealized)``
        and these values are collected and returned.
    printprogress : bool, optional
        If ``True``, progress information is printed while processing
        the loader.
    """

    model.eval()
    correct = 0
    total = 0

    if secure:
        PDmeansarr = []
        PDrealizedarr = []

    if TORCH_AVAILABLE:
        context = torch.no_grad()
    else:
        class _NoGrad:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc, tb):
                pass
        context = _NoGrad()

    with context:
        for images, labels in loader:
            if not secure:
                outputs = model(images)
            else:
                outputs, PDmeans, PDrealized = model(images)
                PDmeansarr.append(PDmeans)
                PDrealizedarr.append(PDrealized)

            if TORCH_AVAILABLE:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                predicted = [max(range(len(o)), key=o.__getitem__) for o in outputs]
                total += len(labels)
                correct += sum(int(p == l) for p, l in zip(predicted, labels))

            if printprogress:
                print(f"{total} test batches done")

    model.train()

    if not secure:
        return 100 * correct / total
    else:
        return 100 * correct / total, PDmeansarr, PDrealizedarr
