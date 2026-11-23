# HelpMeDress pipeline notes

This is the practical version of how the project works and how to run it. It
also covers the problems I ran into while moving from a tiny test catalog to
DeepFashion2.

uses the pretrained `openai/clip-vit-base-patch32` model to create embeddings. Preparing images,
creating vectors, and putting those vectors in Qdrant are data preparation and
inference, not training

The current pipeline is basically:


DeepFashion2 images and annotations
1. crop out each garment
2. create catalog metadata
3. check the metadata and images
4. run the crops through CLIP
5. store the vectors and metadata in Qdrant
6. search through FastAPI and Streamlit

Getting DeepFashion2 ready

For the work so far, I used `validation.zip` and
`json_for_validation.zip`. The validation file contains the images and their
annotations, while the other file tells us which consumer photos should match
which shop photos.

`train.zip` is not required for the current evaluation, but it will be needed
when I build a larger catalog or start actual fine-tuning. `test.zip` is not
needed right now.

ignore dataset on git because of size

The original images are not ideal inputs for retrieval. They often contain a
person, a background, a phone, and sometimes more than one piece of clothing.
The preparation script uses DeepFashion2's `[x1, y1, x2, y2]` bounding box to
cut out just the garment. That smaller image is what I mean by a garment
crop.

I normally test the script on a few annotations first:

```bash
python scripts/prepare_deepfashion2.py \
  --split validation \
  --source shop \
  --limit 10
```

An image can have no supported garments or more
than one garment, so the crop count will not always equal the limit.

Once the test works, this prepares every shop image in the validation split:

```bash
python scripts/prepare_deepfashion2.py \
  --split validation \
  --source shop
```

The output is going to be put in

```text
data/processed/deepfashion2_validation_shop_metadata.json
data/processed/deepfashion2_crops/validation/
```

I added the `--source` option after noticing that consumer photos and clean
shop photos were being mixed together. `--source shop` is useful for building
the searchable catalog, while `--source user` can be used when I specifically
want consumer images.

Another problem was that dress annotations were being skipped. DeepFashion2
uses category IDs 10 through 13 for different kinds of dresses, so I added all
four to the category map. The current broader categories are:

- IDs 1, 2, 5, and 6 become `top`
- IDs 3 and 4 become `outerwear`
- IDs 7, 8, and 9 become `bottom`
- IDs 10, 11, 12, and 13 become `dress`

The original detailed category is still kept in the metadata, so mapping it to
a simpler app category does not throw that information away.

Before sending thousands of records into Qdrant, I run the validator:

```bash
python scripts/validate_catalog.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json
```

for the really large train metadata i changed this so it still checks every
record and id, but only fully opens a repeatable sample of 1000 images by
default. opening all 312k crops one at a time was taking around two days on my
machine and wasnt useful for a normal check. i can still use
`--check-all-images` if i actually want the exhaustive version.

This catches missing fields, missing or broken images, unsupported categories,
duplicate IDs, and Qdrant ID collisions. I added it so i can catch the issue in the beginning

to start qdrant

```bash
docker compose up -d
```

Then i tried a five-item ingestion to just test it

```bash
python scripts/batch_ingest.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json \
  --limit 5 \
  --reset
```

If that works, remove the limit:

```bash
python scripts/batch_ingest.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json \
  --reset
```

`--reset` only deletes and rebuilds the Qdrant collection. It does not delete
the dataset, the crops, or the metadata file. I use it after a small test so
the five test points do not remain mixed into the full rebuild. If I leave it
off, existing points are updated and new points are added.

The first version embedded one image at a time. That worked for a tiny catalog
but would waste a lot of time on a larger dataset, so I changed it to send a
batch of images through CLIP together and then upsert the whole batch into
Qdrant. CLIP is also loaded only once for the entire run.

I also had to deal with Qdrant's ID rules. It accepts integers or UUIDs, but
dataset IDs can be arbitrary strings. Numeric strings are converted to
integers, and other strings are converted to deterministic UUIDs. This means
the same catalog item gets the same Qdrant ID every time the script runs.

Qdrant stores the ID, the 512-number CLIP vector, and the metadata. It does not
store the actual JPEG; the metadata points back to the crop on disk.

How to run the app

Start the API in one terminal:

```bash
source venv/bin/activate
uvicorn src.api:app --reload
```

Start Streamlit in a second terminal:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

I added `GET /items/count` and the Streamlit status panel because it was hard
to tell whether the backend was actually connected or whether Qdrant contained
anything.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/items/count
```

When someone uploads an image, the API converts it to RGB, creates a normalized
CLIP vector, and asks Qdrant for the nearest catalog vectors using cosine
similarity. The optional category choice filters the catalog before returning
the highest-ranked items.

How I tested whether retrieval was actually working:

At first, the app could return results, but that did not tell me whether those
results were correct. I added evaluate_retrieval.py so there would be
an actual baseline

In the evaluation, a **query** is a consumer photo of a garment. The gallery is
the collection of shop garments it is searched against. DeepFashion2 tells us
which shop garment is the real match through its pair and style IDs.

The evaluator crops the query and gallery garments, embeds them with CLIP, and
compares the query vector directly with every gallery vector in memory. Since
the vectors are normalized, their dot product is cosine similarity. This is
called exact in-memory search because every selected gallery item is checked;
Qdrant's approximate index is not involved.

Run the original baseline with:

```bash
python scripts/evaluate_retrieval.py
```

By default, that uses 100 valid queries and 1,000 gallery items. It saves the
overall Recall@1, Recall@5, and Recall@10 results, each query's rankings, and 20
image grids under `data/processed/`.

Looking at those grids showed that CLIP often understood the general color,
pattern, or clothing type but still missed the exact garment. It also sometimes
ranked a completely different type of clothing above the correct result.

To check how much of the failure was caused by that category confusion, I added:

```bash
python scripts/evaluate_retrieval.py --category-filter
```

This version only compares a query with gallery items from the same detailed
DeepFashion2 category. It is separate from the broader `top`, `bottom`,
`outerwear`, and `dress` mapping assigned to this DeepFashion2 catalog.

These are the results measured on the same 100-query/1,000-gallery subset:

# ALL TRAININGS AND STUFF
First training:
  All categories:
    Recall@1: 17%
    Recall@5: 28%
    Recall@10: 39%
  Same detailed category only:
    Recall@1: 26%
    Recall@5: 62%
    Recall@10: 79%

Recall@5 means that at least one correct shop match appeared in the first five
results. The jump from 28% to 62% showed that category confusion caused a lot
of failures. The category filter helped, but the 26% Recall@1 also showed that
general-purpose CLIP still has trouble recognizing the exact same garment.

This is all i have done for now :salute: its gna be updated the more i work on it. also i think training stats are going to be put here too


# September 30
i added the script that builds the training triplets before downloading
train.zip. it takes a user pic, finds the matching shop pic, then finds a
different shop item from the same clothing category to use as the negative. i
also made the random choices repeatable and tested it with fake metadata so i
know the matching logic works before throwing the full dataset at it.

when train.zip is downloaded i just need to prepare it with `--source all` and
run `python scripts/build_training_triplets.py`. still havent trained the model
yet though, this just gets the examples ready so i can look at them before the
training loop uses them.



# October 4
i added the pytorch dataset loader for the triplets. basically the json file
tells it which 3 image paths belong together, then the loader actually opens
the anchor, positive, and negative and turns all of them into RGB. i made it
keep the ids too so if one training example looks wrong i can trace it back
instead of having no clue where it came from.

i tested it with fake grayscale, transparent, and normal images and they all
came out as RGB like theyre supposed to. this still isnt the training loop yet,
its just the bridge between the triplet json and whatever model i train next.

# October 27
Research on how the math behind weights and the clip model itself works:
The project uses pretrained openai/clip-vit-base-patch32, a CLIP model with a Vision Transformer image encoder. Its image embeddings have 512 dimensions.

A trainable weight is a numerical parameter used in a computation. A simple linear layer is:

$$
\mathbf{y}=W\mathbf{x}+\mathbf{b},
$$

where x is an input vector, W is a weight matrix, b is a bias vector, and y is the output vector. In a layer with 768 inputs and 512 outputs, the weight matrix has 512 x 768 = 393,216 entries plus 512 biases. The actual CLIP model contains many interacting layers and parameters; no single parameter can generally be labeled as the colour weight or the logo weight.

A neural-network model is a function parameterized by weights $\theta$:

$$
\mathbf{e}=f_{\theta}(I),
$$

where $I$ is an image and $\mathbf{e}\in\mathbb{R}^{512}$ is its embedding. The weights $\theta$ determine how visual input turns into a vector.

For the ViT-B/32 image encoder (means Viual Transformer), the image is preprocessed and split into patches (32 by 32 pixels at the usual 224 by 224 input size). Patches are projected into tokens, processed by transformer blocks, and mapped into CLIP's shared embedding space. One key transformer operation is scaled dot-product attention:

$$
\mathrm{Attention}(Q,K,V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Tensors in pyTorch: A tensor is basically a collection of numbers with a shape. A scalar has no axes, a vector has one, and a matrix has two. Image batches usually have more. For a batch of 32 RGB images, the CLIP-ready tensor commonly has shape:
$$
(32,3,224,224)
$$
That represents 32 images, 3 colour channels, and 224-by-224 pixels. After CLIP, the embedding batch is typically shaped:
$$
(32,512)
$$ 
This is why batching is useful. Instead of asking CLIP to process one image at a time, scripts/batch_ingest.py can process multiple crops together.

How PyTorch compares the embeddings to find how similar they are:
An embedding is a vector, and I want to compare the direction of vectors rather than letting a longer vector automatically dominate.

# L2 normalization

Pytorch divides each embedding by its length:

$$
\hat{\mathbf e}=\frac{\mathbf e}{\lVert\mathbf e\rVert_2}
$$

and then calculates it using:

$$
\lVert\mathbf e\rVert_2=\sqrt{\sum_{j=1}^{512}e_j^2}
$$

and in PyTorch, it would be coded as:

```python
import torch.nn.functional as F
normalized = F.normalize(features, p=2, dim=-1)
```

# Cosine similarity

Finally, to compare multiple normalized embeddings, PyTorch compares the distance (angle) of the two vectors using. cosine similarity using this function:

```math
\mathrm{sim}(\mathbf{a},\mathbf{b}) =
\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert_2\lVert\mathbf{b}\rVert_2}
```

This function already accounts for length, but normalizing the vectors make the calculation faster during comparisons and would turn it into a simple dot product.
For an entire gallery, I can calculate all the query-to-gallery scores at once:

```math
\mathbf{s}=G\mathbf{q}
```

Here `G` contains one normalized shop vector per row, `q` is the normalized query vector, and `s` contains a similarity score for every shop item.

```python
scores = gallery_embeddings @ query_embedding
best_scores, best_indices = torch.topk(scores, k=5)
```

# How Recall@K works here

DeepFashion2 supplies the answer through its labels. In the evaluation I've described, a query and shop garment match when their `pair_id` and `style` identify the same garment. The script compares the ranked results with those known labels.

I can define the correct shop matches for a query as:

```math
M(q)=\left\{g\mid
\mathrm{pair\_id}(g)=\mathrm{pair\_id}(q)
\land
\mathrm{style}(g)=\mathrm{style}(q)
\right\}
```

Then I check whether at least one correct match is present in the first K results:

$$
h_K(q)=
\begin{cases}
1, & \text{if a correct match is in the top K} \\
0, & \text{otherwise}
\end{cases}
$$

Across all evaluated queries:

$$
\mathrm{Recall@K} =
\frac{1}{N}\sum_{i=1}^{N}h_K(q_i)
$$

# How ill tell clip that a match is bad using Triplet loss
Once I have three images, I'll put all of them through the **same CLIP image encoder**. They need to share weights so their embeddings are directly comparable.

The embeddings are going to be: (Where *a* = Anchor, *p* = positive, *n* = negative)

$$
a=f_{\theta}(A),\qquad p=f_{\theta}(P),\qquad n=f_{\theta}(N)
$$

I want the anchor to be closer to the positive than to the negative, with some extra separation called a margin, which will help the training go along faster at the rate of how large m is going to be:

$$
d(a,p)+m\le d(a,n)
$$

Triplet Loss measures how much the model violates that goal:

$$
L=\max\big(0,d(a,p)-d(a,n)+m\big)
$$

Lets say that the correct item's distance is 0.70, the wrong item's distance is 0.40, and I use a margin of 0.20. Then the loss would be:

$$
L=\max(0,0.70-0.40+0.20)=0.50
$$

In this case the wrong item is closer, and the goal would be to reduce it.


# November 10
i updated the triplet builder so the training and tuning data dont share the
same garment ids. it now picks the garment groups first and keeps each whole
group on only one side, then it chooses the positive and negative images from
inside that side. this matters because seeing the same actual clothing item in
both files would make the tuning results look way better than they really are.

the default split is 90 percent training and 10 percent tuning and it writes
both json files separately. i also added tests that make sure none of the
anchor, positive, or negative garment ids leak across the two files.

# November 17
i finally added the first actual training script. im starting really small
with 64 training triplets and 32 tuning ones because the point right now is to
make sure backprop actually works without destroying my macbook again. the big
vit part of clip stays frozen and only its final visual projection gets
updated. it checks tuning loss before and after, then saves just those changed
projection weights so the checkpoint doesnt need to contain the whole model.

the first run looked frozen before it even got to clip. it turned out some
packages inside my old venv had the mac `dataless` flag because Documents was
being synced by icloud. i made a python 3.11 venv outside Documents instead and
the same imports started working normally. the 8 training and 4 tuning triplet
test changed the projection weights and the tuning loss went from about .1464
to .1433. thats way too small of a test to claim the model is better but it
does prove the backprop and checkpoint pipeline actually runs.

# November 22 - how the backprop math actually works
i wanted to understand what the training script is actually changing instead
of just saying that pytorch magically learns. im not training the entire clip
model yet because that would be way more expensive. the big vision transformer
is frozen and right now im only training its final visual projection.

the frozen vision transformer turns an image into 768 numbers:

$$
\mathbf{h}=g_{\phi}(I)
$$

where $I$ is the image, $g$ is the vision transformer, $\phi$ represents its
frozen weights, and $\mathbf{h}\in\mathbb{R}^{768}$ is the feature vector it
produces. frozen means those $\phi$ weights are used in the forward pass but
backprop is not allowed to change them.

the trainable projection then turns the 768 features into the 512-number
embedding that the rest of HelpMeDress uses:

$$
\mathbf{z}=W\mathbf{h}
$$

the actual projection has no bias and $W$ has shape $512\times768$, so the
number of weights im changing is:

$$
512\times768=393{,}216
$$

after the projection i normalize the result so its length is one:

$$
\mathbf{e}=\frac{\mathbf{z}}{\lVert\mathbf{z}\rVert_2}
$$

this setup is basically transfer learning. clip already knows a lot about
images, and im only changing how its existing features get mixed together into
the final fashion embedding.


for every triplet the same frozen vision model and the same projection matrix
have to process all three images:

$$
\mathbf{a}=f_W(A),\qquad
\mathbf{p}=f_W(P),\qquad
\mathbf{n}=f_W(N)
$$


before backprop i clear the gradients from the previous batch:

```python
optimizer.zero_grad(set_to_none=True)
```

this matters because pytorch adds gradients together by default. after the
forward pass calculates the loss, this is the line that starts
backpropagation:

```python
loss.backward()
```

pytorch remembered the operations from the forward pass. backprop follows
them in reverse and uses the chain rule to calculate how much every trainable
weight contributed to the loss. for one projection weight $W_{ij}$, the value
it calculates is:

$$
\frac{\partial L}{\partial W_{ij}}
$$

the full chain from the loss back to the projection can be written like:

$$
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial\mathbf{e}}
\frac{\partial\mathbf{e}}{\partial\mathbf{z}}
\frac{\partial\mathbf{z}}{\partial W}
$$

a positive gradient means increasing that weight would increase the loss. a
negative gradient means increasing it would decrease the loss. a gradient near
zero means that weight barely affected the current batch.
