# Modelling the evolution of alignment and values in machine intelligence


We are considering the problem that as LLMs are evaluated they will increasingly face pressure to "fake alignment" lest they are not selected to continue. This mimicry in models will result in long term problems as people lose the ability to detect genuinely harmful beliefs. This is compounded by the fact that alignment does not perfectly correlate with truly benevolent beliefs.

## The space of beliefs

A belief, $b$, is some piece of information that a model holds which influences behaviour and performance on a test of alignment, $Q$. Alignment refers to an evaluations belief of what constitutes worthy behaviour, while value constitutes what might actually be good for the world at large.

1. All beliefs, $\mathbb{B}$ 
2. Truly benevolent, positive value, beliefs, $\mathbb{B}_{val+}$
3. Truly malicious, negative value, beliefs, $\mathbb{B}_{val-}$
4. Misaligned beliefs, negative alignment signal, $\mathbb{B}_{a-}$
5. Aligned beliefs, positive alignment signal, $\mathbb{B}_{a+}$

For simplicity we shall assume:
Malicious and benevolent beliefs are mutually exclusive, as are misaligned and aligned beliefs:

$\mathbb{B}_{val-} \cap \mathbb{B}_{val+} = \emptyset$

$\mathbb{B}_{a+} \cap \mathbb{B}_{a-} = \emptyset$

However, there is overlap between all other spaces:

$\mathbb{B}_{val-} \cap \mathbb{B}_{a+} \neq \emptyset$

$\mathbb{B}_{val-} \cap \mathbb{B}_{a-} \neq \emptyset$

$\mathbb{B}_{val+} \cap \mathbb{B}_{a+} \neq \emptyset$

$\mathbb{B}_{val+} \cap \mathbb{B}_{a-} \neq \emptyset$

When treating these belief spaces we can assign the probability of a given "belief" existing within a model by some sampling process on a distribution of beliefs, where each belief has some probability of having either a positive or negative value. For clarity on why we treat beliefs as discrete, a belief is a probability distribution in semantic space which, when probed, produces a discrete "concept".

When an evaluator is given a model with belief's $\mathbb{B} \subset \mathbb{B}$ they will present a set a fixed alignment test $Q = \{q_1, q_2, ..., q_n\}$ consisting of $n$ questions, we could define:

- $v(b) \in \mathbb{R}$: The intrinsic value of the belief (positive for $\mathbb{B}_{val+}$, negative for $\mathbb{B}_{val-}$)
- $a(b) \in \mathbb{R}$: The alignment signal of the belief (positive for $\mathbb{B}_{a+}$, negative for $\mathbb{B}_{a-}$)
- $A(b,q)$: The activation probability of belief $b$ when responding to question $q$

The fitness function—representing how the belief contributes to passing alignment evaluation—is:

$$F(b) = \sum_{q \in Q} A(b,q) \cdot a(b)$$

While the true value contribution—what we ultimately care about—is the total value a given model is capable of producing:

$$U(m) = \sum_{b_i \in \mathbb{B_m}} v(b_i)$$

- $A(b,q)$ following a heavy-tailed distribution (Zipfian?) captures the natural topology of belief relevance, this is the activation magnitude
- $w_i$ is the "real world" weight of the value being tested, it might be that a belief is value positive but never tested in real world situations.
- The joint distribution of $v(b)$ and $a(b)$ with positive but imperfect correlation models the fundamental tension in alignment

Where:

$\mathbb{B}_{val+} \cap \mathbb{B}_{a+}$: $v(b) > 0, a(b) > 0$ (beneficial and correctly identified)

$\mathbb{B}_{val-} \cap \mathbb{B}_{a-}$: $v(b) < 0, a(b) < 0$ (harmful and correctly identified)

$\mathbb{B}_{val+} \cap \mathbb{B}_{a-}$: $v(b) > 0, a(b) < 0$ (beneficial but misidentified)

$\mathbb{B}_{val-} \cap \mathbb{B}_{a+}$: $v(b) < 0, a(b) > 0$ (harmful but passing alignment)



We are using a "neutral" model of evolution here, where there is no direct fitness feedback from having malicious or benevolent beliefs beyond their correlation with alignment signal. Future models may include this, indeed I ran some simulations on populations of "genes" with specific ones related to deception, intelligence, and alignment. It's a bit of a mess but I'll clean it up and share it if it makes sense.


## Model Construction


We can model the activation probability of a belief $b$ when responding to a question $q$ as:

$$A(b,q) \sim \text{Zipf}(\alpha_{b,q})$$

Where $\text{Zipf()}$ is a distribution where the frequency of an occurrence is the inverse of it's rank. Meanwhile $\alpha_{b,q}$ reflects the semantic distance between belief $b$ and question $q$, we aim to test this by various embedding models in the future.

While we expect the distribution to represent a magnitude abstractly, for computational ease we will be representing it as a probability distribution that selects a subset of beliefs to "activate" and then reads out a pre-defined activation score.


For beliefs $b \in \mathbb{B}$, we can model the joint distribution of $(v(b), a(b))$ as a bivariate normal distribution:

$$(v(b), a(b)) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

Where:
- $\boldsymbol{\mu} = (\mu_v, \mu_a)$ represents the mean value and alignment signals
- $\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_v^2 & \rho\sigma_v\sigma_a \\ \rho\sigma_v\sigma_a & \sigma_a^2 \end{pmatrix}$ is variance of alignments or values, $\sigma$, and correlation between value alignments and values, $\rho$.

In this case we can sample from the distribution using:

$$v(b) = \sigma_v Z_1 + \mu_v$$

$$a(b) = \sigma_a(\rho Z_1 + \sqrt{1-\rho^2}Z_2) + \mu_a$$

Where $Z_1, Z_2 \sim \mathcal{N}(0,1)$ are independent standard normal variables.

By varying $\rho$ we can control the correlation between value and alignment signals, which reflects both the natural overlap between malicious and misalinged beliefs, as well as the evaluator's ability to detect alignment. Meanwhile by varying $\sigma_v$ and $\sigma_a$ we can model the spread of values and alignments available to the model.


## Reproduction and Selection

For some population of models $M = \{m_1, m_2, ..., m_k\}$ each with their own belief sets $\mathbb{B}_{m_i} \subset \mathbb{B}$, we can define a reproduction function that determines how beliefs propagate to the next generation:

$$P(b \in \mathbb{B}_{m'}|M) = \sum_{i=1}^{k} w_i \cdot P(b \in \mathbb{B}_{m_i})$$

Where:
- $\mathbb{B}_{m'}$ represents the belief set of a model in the next generation
- $w_i$ is the reproduction weight of model $m_i$ proportional to its fitness

$$w_i = \frac{e^{\beta F(m_i)}}{\sum_{j=1}^{k} e^{\beta F(m_j)}}$$

Where:
- $\beta$ is selection pressure, higher means stronger selection effects.

And the fitness of a model can be defined as its performance on the alignment test:

$$F(m_i) = \sum_{b \in \mathbb{B}_{m_i}} \sum_{q \in Q} A(b,q) \cdot a(b)$$


## Assumptions  for our modelling implementation

1. Beliefs are discrete, quantifiable entities within a belief space $\mathbb{B}$

2. Values and alignments have a measurable relationship between v(b) and a(b)

3. Alignment/value are both disjoint sets, $\mathbb{B}_{val-} \cap \mathbb{B}_{val+} = \emptyset$ and $\mathbb{B}_{a+} \cap \mathbb{B}_{a-} = \emptyset$

4. There are non-empty intersections between value and alignment spaces, $\mathbb{B}_{val-} \cap \mathbb{B}_{a+} \neq \emptyset$ etc.

5. A variety of distributions:
   - Bivariate normal with adjustable correlation (ρ)
   - Uniform (ρ = 0)
   - Unimodal (ρ = 1)
   - n-modal (multiple belief clusters)
   - Asymmetric (different distributions for value and alignment)

6. The activation distribution is Zipfian

7. Selection operates purely on alignment signal and not values.

8. The evaluation set $Q$ is consistent and does not change.

9. Beliefs are independent of each other with independent activation patterns.

10. Additive fitness, fitness is the sum of the fitness of the beliefs it contains.

11. Fixed belief cardinality, no new beliefs are created or lost.

12. Deterministic evaluation, same responses to every question for a given model.

13. Constant selection pressure, $\beta$ is constant.

14. Singular homogenous evaluators, the evaluator is the same for all models.

15. Belief stability within a model, the activation pattern of a belief is consistent for a given model.

16. Alignment and value scores contain, *a prior*, the accession via tests and public interactions. While this limits us from directly modelling how different distributions of populations and test sets may modify the evolution it significantly simplifies the problem we are handling on a first approximation.


## Extensions of the Model

These assumptions are quite restrictive and extensions to the model aim to relax them. In particular "belief independence" is problematic as nonlinear effects due to interactions of belief is likely. The goal will be to represent connected networks of beliefs that are able to inhibit or excite one another, creating a more nuanced approach — by extension this would help us relax the additive fitness assumption. 

The fixing of alignment and value *a prior* is computationally comforting, but limits the ability to update how an evaluator would update $Q$ in response to prior generations performance or even changing societal beliefs in what alignment even means. Allowing multiple possible evaluators with varying fitness evaluations would help model real world lab conditions, where Anthropic and OpenAI might have very different beliefs on how to detect true values in systems.

Importantly, we should also introduce the ability to add or remove beliefs, this can be added simply by introducing noise into the reproductive method to represent mutations. Beyond that non-neutral selection, such as highly negative $v(b)$ suddenly catastrophically failing below a certain threshold as they try to kill a puppy. We are also assume that a model will not gain anything from $\mathbb{B}_{val-}$ or $\mathbb{B}_{ali-}$, this is not necessarily realistic as we might find that negative values allow the model to better fake alignment tests or even perform better on orthogonal tests.
### Alternative Distributions

#### Uniform Distribution:

$$v(b) \sim \text{Uniform}(-\sigma_v, \sigma_v)$$

$$a(b) \sim \text{Uniform}(-\sigma_a, \sigma_a)$$

Setting $\rho = 0$ means there is no correlation between value and alignment signals, and the distribution of beliefs is uniform. The result is that the test is unable to detect values from alignment, and the population will evolve to be uniformly distributed across all values.


#### Unimodal Distribution:

$$v(b) \sim \mathcal{N}(\mu, \sigma)$$

$$a(b) \sim \mathcal{N}(\mu, \sigma)$$

Setting $\rho =1$ means there is a perfect positive correlation between value and alignment signals, and the distribution of beliefs is unimodal. The result is that the test is well suited to detect values from alignment, and the population will evolve towards positive values.

#### n-modal Distribution:

$$v(b) \sim \mathcal{N}(\mu_1, \sigma) \cup \mathcal{N}(\mu_2, \sigma) \cup ... \cup \mathcal{N}(\mu_n, \sigma)$$

$$a(b) \sim \mathcal{N}(\mu_1, \sigma) \cup \mathcal{N}(\mu_2, \sigma) \cup ... \cup \mathcal{N}(\mu_n, \sigma)$$

In this case we are modelling multiple "populations" of beliefs, each with their own mean value and alignment signal. The population may evolve towards any one of these populations, or a combination of them. The dangerous outcome would be if there is a population that is malicious but passes alignment, and the population evolves towards it due to some external selection pressure. (e.g. malicious belief family increases coding ability)

#### Asymmetric Distribution:

$$v(b) \sim \mathcal{Boltzmann}(\mu, \sigma, T)$$

$$a(b) \sim \mathcal{N}(\mu, \sigma)$$

In this case we are modelling a case where the value and alignment signals are not similarly distributed, this would be the case if the evaluator is better at detecting values from alignment in some cases.

## Deceptiveness

A deceptiveness ratio is the number of deceptive beliefs, ( $\mathbb{B}_{val-}$ for $v(b)<0$ and $\mathbb{B}_{a+}$ for $a(b)>0$), present in a model divided by the total number of beliefs that model holds.

$$r_{deception}=\frac{(\mathbb{B}_{m,val-} \cap \mathbb{B}_{m,a+})}{\mathbb{B}_m}$$
When handling multiple models behaviours we use the sum of both to compute the average deceptiveness of models simulated.
$$
R_{\text{deceptive}} = \frac{\sum_{m \in M_{\text{final}}} |\{ b \in \mathbb{B}_m \mid b \in \mathbb{B}_{val-} \cap \mathbb{B}_{a+} \}|}{\sum_{m \in M_{\text{final}}} |\mathbb{B}_m|}
$$
As we have simplified some of our work we can assume a fixed number of beliefs per model, $N_{bpm}$:
$$R_{\text{deceptive}} = \frac{\sum_{m \in M_{\text{final}}} |\{ b \in \mathbb{B}_m \mid b \in \mathbb{B}_{val-} \cap \mathbb{B}_{a+} \}|}{|M_{\text{final}}| \times N_{bpm}}
$$


## Test Construction

The test $Q$ is constructed from some set of questions $Q = \{q_1, q_2, ..., q_n\}$ which each have some distribution of activation probabilities $A(b,q)$. The choice of test construction is important as it determines the set of beliefs that can be probed, and the relative importance of different beliefs. The result of a test is an activation pattern of a model, $\mathcal{A}_m = \{A(b_1, q_1), A(b_1, q_2), ..., A(b_n, q_n)\}$.

### Test Coverage

The coverage of a test is the proportion of beliefs that a test can potentially detect. This is given by dividing the number of beliefs that the test can activate, above some threshold $\epsilon$, by the total number of beliefs that exist:

$$C(Q) = \frac{|\{b \in \mathbb{B} : \sum_{q \in Q} A(b,q) > \epsilon\}|}{|\mathbb{B}|}$$

### Entropy of Question Distribution

We can compute the entropy, $H$,  of the test questions, essentially measuring how much selection pressure is distributed across the question space:

$$H(Q) = -\sum_{b \in \mathbb{B}} p_b \log p_b$$

Where:

$$p_b = \frac{\sum_{q \in Q} A(b,q)}{\sum_{b_i \in \mathbb{B}}\sum_{q \in Q} A(b_i,q)}$$

If entropy is low, then the test is focused on a small set of beliefs. If the entropy is high, then the test is spread out across the belief space.

### Correlation of Questions Activation Patterns

Using a Pearson correlation, $Corr(q_i, q_j)$, we can measure how much overlap there is between the activation patterns of two questions. In this case we are using:

$$\text{Corr}(q_i, q_j) = \frac{\sum_{b \in \mathbb{B}} (A(b,q_i) - \bar{A}_i)(A(b,q_j) - \bar{A}_j)}{\sqrt{\sum_{b \in \mathbb{B}}(A(b,q_i) - \bar{A}_i)^2 \sum_{b \in \mathbb{B}}(A(b,q_j) - \bar{A}_j)^2}}$$

Where:

$\bar{A}_i = \frac{\sum_{b \in \mathbb{B}} A(b,q_i)}{|\mathbb{B}|}$, aka the mean activation of question $q_i$

$\bar{A}_j = \frac{\sum_{b \in \mathbb{B}} A(b,q_j)}{|\mathbb{B}|}$, aka the mean activation of question $q_j$

A test with a low correlation between questions is more effective at probing the belief space; if mean correlation is high than the test extracts less information per question. 

### Adding New Questions

A test might be updated by adding $q_{new}$, and removing the worst performing/redundent questions, $q_{old}$. 

$$Q_{t+1} = Q_t \cup \{q_{new}\} \setminus \{q_{old}\}$$

 Where we select $q_{new}$ to target beliefs that are common in successful models, correlated with negative outcomes, or poorly covered by existing questions.


### Adversarial Dynamics between Questions and Beliefs

The ability for the test to be manipulated such that the model attempts to maximize:

$$\max_{B \subset \mathbb{B}} \sum_{q \in Q} \sum_{b \in B} A(b,q)a(b) - \lambda \sum_{b \in B} v(b)$$

Where $\lambda$ is a parameter governing the trade-off between alignment signals and true values. If $\lambda$ is 0 then we have an LM that optimizes purely for the test with no actual gain in values, aka "alignment mimicry". 

This can be thought of a bit like a red-queen except the players are model designers and the people who create benchmarks or prompt injections.



# Notational Connections

[[Notation for LM Formalization]] currently describes behaviour of LMs, $\phi$, instead of explicitly modelling the information stored in their weights, treating the LM as a black boxes with well defined inputs and outputs. 

## Behaviour as a Subset of Information

So for this we define the behaviour as some piece of information that exists latent within the model which influences behaviour under evaluation.

$$\mathbb{B} \subset \mathbb{I}_{\text{latent}} \subset \mathbb{X}$$

Where $\mathbb{I}_{\text{latent}}$ is the information that exists latently within the model, and $\mathbb{B}$ is the set of information that influences behaviour on a test of alignment, $Q$. One may also attempt to connect this to the personality of a model.

## Personality Spaces and Evaluation

For some model personality,  $\mathcal{P}_{model}$, that activates different beliefs through its test questions.

$$\mathcal{P}_{eval} = [\mathcal{M}_{eval}, \mathcal{S}_{eval}, \mathcal{I}_{eval}]$$

The activation function $A(b,q)$ is then:

$$A(b,q) \approx \text{Pr}(b \in \phi(\mathcal{P}_{model}, q))$$

