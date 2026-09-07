# Introduction: panel formation before blind evaluation

## Blind evaluation begins before the estimator

Evaluating decision-makers without an answer key is a recurring problem: in
crowdsourcing, in ensembles of classifiers, in deliberative bodies, and in any
setting where the truth is expensive, contested, or unavailable at evaluation
time — the problem Dawid and Skene first formalized for observer error-rate
estimation without ground truth
[Dawid and Skene (1979)](https://doi.org/10.2307/2346806).
Later learning-from-crowds work made the same inferential problem operational for
noisy human labels and missing gold standards
[Raykar et al. (2010)](https://jmlr.org/papers/v11/raykar10a.html), and
budget-optimal crowdsourcing work shows why worker reliability, redundancy, and
task assignment cannot be separated when answers are inferred from noisy repeated
judgments rather than gold labels
[Karger et al. (2014)](https://doi.org/10.1287/opre.2013.1235). A parallel line
estimates classifier accuracy *without any labels* by exploiting the structure of
agreements among multiple predictors — logic- and constraint-based
[@platanios2014] and spectral [@parisi2014] — and, like the exact NTQR solver, these
methods lean on an **error-independence** (or low-rank residual-dependence)
assumption whose violation is exactly the failure mode we make tunable here. The
`ntqr` package (v0.8) frames the problem as algebraic logic for unsupervised
evaluation from unlabeled decision data
[Corrada-Emmanuel (2026)](https://pypi.org/project/ntqr/). Our
generative model for *violating* that assumption — same-group judges sharing a
latent shock through a Gaussian copula on a probit-thresholded competence variable —
is the standard construction for correlated binary outcomes [@emrich1991; @nelsen2006],
which lets us preserve each judge's marginal accuracy exactly while dialing
cross-judge error correlation.
Given the joint pattern of agreements and
disagreements among three binary judges, its error-independent (EIE) evaluator
returns the possible logically consistent combinations of item prevalence and
per-judge accuracy that could have produced those votes — *with no labels at
all*.

The `ntqr` evaluator, however, takes the panel as a fixed input. Its claims are
about which evaluations are logically consistent for a given set of judges. The
judges themselves arrive from
*somewhere*: a hiring process, a volunteer pool, a citizens' assembly lottery, a
top-k leaderboard. That upstream step — **panel formation** — is the part this
project studies. Our central question is deliberately one level above NTQR:

> Across matched synthetic population and corpus settings, how do the *strategy
> that forms the panel* and the *size of the panel* change the oracle-referenced
> error of no-answer-key NTQR-style evaluations?

Two scope notes frame everything below. First, *oracle-referenced recovery error*
is the distance between the blind, no-answer-key estimate and the estimate you would
have obtained *with* the answer key — lower is better. Second, NTQR's exact
error-independent evaluator solves for *exactly three* binary judges, so a panel of
any larger size is evaluated as an ensemble of its constituent trios; "panel size,"
throughout, means how many such trios the panel contributes, not a larger joint
solve.

## Application review as the empirical stress test

Application review is the manuscript's concrete empirical stress test because it
combines expert judgment, scarce labels, panel formation, and plausible nuisance
bias in one setting. Studies of academic and grant peer review emphasize that
expert panels are not transparent measurement devices: judgment is field-situated
[Lamont (2009)](https://doi.org/10.4159/9780674054158), replication of review
decisions can be fragile [Peters and Ceci (1982)](https://doi.org/10.1017/S0140525X00011183),
reviewer agreement can be low even on the same proposals
[Cole et al. (1981)](https://doi.org/10.1126/science.7302566);
[Pier et al. (2018)](https://doi.org/10.1073/pnas.1714379115), and statistical
analysis of NIH review scores shows that panel-scoring uncertainty can materially
change which proposals would be funded
[Johnson (2008)](https://doi.org/10.1073/pnas.0804538105). Productivity follow-up
work adds a separate caution: NIH peer-review percentile scores can be weak
predictors of grant productivity
[Fang et al. (2016)](https://doi.org/10.7554/eLife.13323),
and peer-review bias is a documented design concern rather than an exotic failure
mode [Lee et al. (2013)](https://doi.org/10.1002/asi.22784);
[Tomkins et al. (2017)](https://doi.org/10.1073/pnas.1707323114);
[Helmer et al. (2017)](https://doi.org/10.7554/eLife.21718). The postdoctoral
fellowship version is also historically apt: Wennerås and Wold's analysis of
postdoctoral fellowship review found nepotism and sexism in peer review
[Wennerås and Wold (1997)](https://doi.org/10.1038/387341a0), while broader
funding studies report racial disparities in award outcomes
[Ginther et al. (2011)](https://doi.org/10.1126/science.1196783).

Those literatures motivate the *mechanism* we stress-test, not the conclusion.
This project uses fictitious postdoctoral applications and synthetic age metadata
so the hidden quality label is generated independently of age. Age is included as
a protected-attribute nuisance axis because age bias and age-discrimination
effects are documented in employment-relevant settings
[North and Fiske (2013)](https://doi.org/10.1177/0146167213480043);
[Neumark et al. (2019)](https://doi.org/10.1086/701029). The manuscript therefore
asks a controlled design question, not a policy question: if reviewer expertise
and irrelevant age bias are known in the generator or prompt profile, which
sampling rule best limits oracle-referenced NTQR error and age-conditioned
recommendations?

## Sortition from civic lotteries to evaluator sampling

Sortition — selection by lottery, often with quotas that make the drawn body
mirror the population — is the canonical *fair*, non-comparative panel-formation
rule
[Stone (2011)](https://doi.org/10.1093/acprof:oso/9780199756100.001.0001).
Modern implementations use auditable maximin algorithms
[Flanigan et al. (2021)](https://doi.org/10.1038/s41586-021-03788-6) that
maximize the minimum selection probability subject to representativeness
constraints. Sortition is attractive precisely because it does **not** select on
competence; it selects on representativeness. That makes it a sharp test case for
NTQR: a representative panel is heterogeneous and, by design, *not* curated for
accuracy. Does representativeness help or hurt an estimator that depends on the
statistical independence of judges' errors?

That procedural tension has a long pre-1800 lineage. Aristotle's political
theory treats lot and election as constitutional signals — lot as democratic,
election as oligarchic — while the Athenian institutional account describes
juries and offices allocated by lot
[Aristotle (Politics)](https://topostext.org/work/100);
[Aristotle (Athenian Constitution)](https://topostext.org/work/99). Medieval and
early-modern writers kept the same problem visible under different vocabularies:
[Aquinas (Summa Theologiae II-II q.95 a.8)](https://www.newadvent.org/summa/3095.htm#article8)
distinguished practical uses of lots from divinatory misuse, and
[Contarini (1599)](https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp14764)
described Venetian mixed selection machinery as an anti-factional civic design.
Enlightenment writers then made the lot-versus-choice contrast explicit again:
[Montesquieu (1748)](https://oll.libertyfund.org/titles/montesquieu-complete-works-vol-1-the-spirit-of-laws)
and [Rousseau (1762)](https://oll.libertyfund.org/titles/cole-the-social-contract-and-discourses)
both distinguish election by lot from election by choice. We cite these sources
as procedural history, not as direct empirical precedent: this manuscript does not
claim that Athenian juries, Venetian offices, or scholastic accounts of chance
anticipate NTQR. They do show that the upstream choice between lot, choice,
status, and asserted competence is an old institutional design problem.

Randomness is already a serious proposal in adjacent research-funding design, but
usually at a different point in the pipeline: collective-allocation and
modified-lottery proposals address how funds might be allocated after review or
thresholding [Bollen et al. (2014)](https://doi.org/10.1002/embr.201338068);
[Fang and Casadevall (2016)](https://doi.org/10.1128/mBio.00422-16). Lottery
arguments also arise from the maverick-science problem: if high-variance projects
are hard to rank reliably, randomized allocation can be an epistemic risk-control
device rather than only an administrative convenience
[Avin (2019)](https://doi.org/10.1016/j.shpsa.2018.11.006). This
manuscript moves the lottery upstream. It asks how reviewer *sampling* changes an
unlabeled evaluator before any funding decision is made.

The tension is not artificial. Classical jury-theorem results make group accuracy
depend on competence, independence, and aggregation rule
[Grofman et al. (1983)](https://doi.org/10.1007/BF00125672), while diversity
results show that heterogeneous problem-solving groups can outperform
ability-selected groups under specific search conditions
[Hong and Page (2004)](https://doi.org/10.1073/pnas.0403723101). Deliberative
public-consultation work likewise treats representative participation as a
normative and epistemic design choice, not just a sampling convenience
[Fishkin (2009)](https://global.oup.com/academic/product/when-the-people-speak-9780199604432).
The formal voting-theory lineage is also historical rather than merely modern:
[Borda (1781)](https://bibbase.org/network/publication/denbspborda-mmoiresurleslectionsauscrutin-1781)
proposed a scored ballot for elections, and
[Condorcet (1785)](https://archive.org/details/bub_gb_RzAVAAAAQAAJ) analyzed the
probability of correct plurality decisions. Those works mostly take the voters or
jurors as given. The present instrument asks the upstream question they leave
exogenous: how does the rule that forms the evaluator panel change the blind
recovery problem before aggregation begins?
This manuscript does not assume which rationale wins under NTQR; it measures the
trade-off in a controlled binary-evaluation instrument.

We compare four panel-formation strategies against each other and against the
supervised oracle: auditable representative sortition, uniform random selection
(the honest baseline), single-bloc ideological selection (a deliberately
correlated, non-representative comparator), and competence-first expertise
thresholding. The oracle and the strong baselines are first-class comparators, not
strawmen — the point of the instrument is to *measure*, including measuring our own
preferred narrative against an honest null.

## Falsifiable claims and negative controls

We state the study as five falsifiable hypotheses (H1–H5) and let the regenerated
artifacts adjudicate each. The synthetic deterministic track tests H1–H4 against a
known oracle; the live single-model companion tests H5. Methods (Table
@tbl:falsification) states the load-bearing assumption and the negative-control
check behind each, and the Discussion returns an explicit verdict hypothesis by
hypothesis.

1. **H1 — Formation strategy is the dominant lever.** Different panel-formation
   rules yield materially different oracle-referenced EIE recovery error, even at a
   fixed population and panel size. *Tested by* the weighted-mean strategy ranking
   with a bootstrap separation gate and a power budget ([@fig:ranking]).
2. **H2 — Concentrating correlated bias degrades recovery.** Single-bloc selection,
   which seats judges whose errors are correlated, recovers with higher EIE error
   than a representative draw, because the error-independence assumption NTQR rests
   on is most stressed by a correlated panel. *Tested by* the
   representative-minus-single-bloc contrast across the full regime grid against
   analytical sign predictions ([@fig:repideo]), and resolved by the
   composition-coupled error confound that fans the strategies apart as within-bloc
   coupling rises ([@fig:blocphase]).
3. **H3 — Size is a sampling knob, not guaranteed improvement.** Forming a larger
   ensemble gives more trios to average over, but whether that helps or hurts
   recovery — and whether any effect comes from the larger ensemble violating the
   solver's error-independence assumption more — is an empirical question; the power
   question is how many observations at the analyzed grain would be needed to
   resolve a contrast of the observed magnitude. *Tested by* the per-strategy size
   sweep, a paired regime-controlled size contrast, a per-trio conditioning
   diagnostic, and the power/MDE budgets ([@fig:powercurve], [@fig:powervn],
   [@fig:triocond]).
4. **H4 — Error-correlation is measurable, and recovery degrades with it.** A
   controlled correlation injection produces a realized error-correlation that NTQR
   itself reports, and oracle-referenced recovery error should rise as that
   realized correlation grows. *Tested by* the tolerance sweep, reporting the
   realized-correlation trend and an OLS recovery-vs-correlation slope with a
   bootstrap interval ([@fig:tolerance]); the recovery slope, unresolved by the
   global-injection sweep, is resolved positive once the correlation is
   composition-coupled and marginal-preserving ([@fig:blocphase]).
5. **H5 — The synthetic ranking transfers to a live single-model panel.** The
   strategy ordering measured against the synthetic oracle reproduces when one
   local `{{POSTDOC_MODEL}}` model is prompted as different reviewers. *Tested by* a
   matched-grain cross-track ranking comparison and cell-level directional
   alignment ([@fig:trackinversion], [@fig:postdocalign]).

These hypotheses are written to be refutable, and the data refute several: H3 and
H5 are rejected; H2 and H4 are unresolved on the baseline grid and resolve only once
correlation is coupled to panel composition; and H1's ranking collapses to
competence-first versus a bunched remainder. The instrument, its resolved cells, its
explicit design-limited cells, and the axis-conditional negative control that keeps
the sortition result honest — not a manufactured sortition win — are the contribution.
