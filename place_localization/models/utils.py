from pytorch_metric_learning import losses, miners, distances


def get_distance(distance_name: str) -> distances.BaseDistance:
    match distance_name:
        case 'EuclideanDistance':
            return distances.LpDistance(normalize_embeddings=True, p=2)
        case _:
            raise NotImplementedError(f'Unsupported distance: {distance_name}')


def get_loss_function(loss_name: str, distance: distances.BaseDistance, num_classes: int,
                      embedding_size: int) -> losses.BaseMetricLossFunction:
    match loss_name:
        case 'ArcFaceLoss':
            return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, distance=distance)
        case 'CosFaceLoss':
            return losses.CosFaceLoss(num_classes=num_classes, embedding_size=embedding_size, distance=distance)
        case 'TripletMarginLoss':
            return losses.TripletMarginLoss(distance=distance)
        case _:
            raise NotImplementedError(f'Unsupported loss: {loss_name}')


def get_miner(miner_name: str, distance: distances.BaseDistance) -> miners.BaseMiner:
    match miner_name:
        case 'BatchHardMiner':
            return miners.BatchHardMiner(distance=distance)
        case 'MultiSimilarityMiner':
            return miners.MultiSimilarityMiner(distance=distance)
        case 'TripletMarginMiner':
            return miners.TripletMarginMiner(distance=distance)
        case _:
            raise NotImplementedError(f'Unsupported miner: {miner_name}')