import pandas as pd

# !pip install git+https://github.com/raphaelsty/mkb --upgrade
from mkb import utils
from mkb import datasets
from mkb import models
from mkb import losses
from mkb import sampling
from mkb import evaluation
from mkb import compose

import torch

df = pd.DataFrame({
    'user': [1, 2, 3, 4, 5],
    'banque': ['Societe Generale', 'Credit Lyonnais', 'Chinese National Bank', 'Chinese National Bank', 'QIWI'],
    'country': ['France', 'France', 'China', 'China', 'Russia']
})

# Keys allows to create a graph such as:
# [user_1, banque_1]
# [user_2, banque_1]
# [user_3, banque_2]
# [user_3, country_1]
keys = {
    'user': ['banque', 'country'],
}

# Avoid collision between entities, eg, distinguish banque 1 with user 1.
prefix = {column: f'{column}_' for column in df.columns}

train = utils.dataframe_to_kg(
    df=df,
    keys=keys,
    prefix=prefix
)

_ = torch.manual_seed(42)

# Set device = 'cuda' if you own a gpu.
device = 'cuda'

dataset = datasets.Dataset(
    train=train,
    valid=train,
    batch_size=256,
)

model = models.TransE(
    entities=dataset.entities,
    relations=dataset.relations,
    gamma=9,
    hidden_dim=500,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.00005,  # 0.00005
)

negative_sampling = sampling.NegativeSampling(
    size=256,
    train_triples=dataset.train,
    entities=dataset.entities,
    relations=dataset.relations,
    seed=42,
)

validation = evaluation.Evaluation(
    true_triples=dataset.true_triples,
    entities=dataset.entities,
    relations=dataset.relations,
    batch_size=8,
    device=device,
)

pipeline = compose.Pipeline(
    epochs=10,  # 1000 Usually or more
    eval_every=10,  # Depends on epochs
    early_stopping_rounds=5,
    device=device,
)

pipeline = pipeline.learn(
    model=model,
    dataset=dataset,
    evaluation=validation,
    sampling=negative_sampling,
    optimizer=optimizer,
    loss=losses.Adversarial(alpha=0.5)
)


# PCA before concatenation of embeddings:
embeddings = utils.map_embeddings(
    df=df, prefix=prefix, embeddings=model.embeddings['entities'], n_components=2)

# PCA after concatenation of embeddings:
embeddings = utils.row_embeddings(
    df=df, embeddings=model.embeddings['entities'], prefix=prefix, n_components=2)

embeddings.to_csv('embeddings.csv')
