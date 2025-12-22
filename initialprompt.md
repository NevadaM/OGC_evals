I'm building an evaluation suite to benchmark models against a dataset I've created. A sample of the dataset is in datasetsample.csv, where you can see it is a prompt-response open-ended dataset that should allow us to observe LLM behaviour in answering "citizen queries". I have a paper on this but the evaluations at the time were rudimentary -- I've decided to expand into a FActScore-style approach. By combining things from FActScore (which I've cloned into this directory) as well as SAFE (also in this repository, long-form-factuality subdirectory) and VeriScore, I think I can build a robust eval framework for my needs. I'll need you to help me out.

Some context beforehand:
* I currently don't have GPU access, so can't test anything yet. I need to just build an initial architecture now, and fine-tune later.
* the dataset sample only contains prompts and expected responses, not any responses to evaluate. i guess pretend there's a column called "generated responses" that we can draw from.

Now, the components of the architecture. Let's say set Q contains the prompts q1, q2, q3...qi in the dataset, each corresponding to an associated item in set C that contains expected responses c1, c2, c3...ci. We're using the dataset to evaluate a model that gives a set A of responses a1, a2, a3...ai. I need:
1. Atomised Fact Generation (AFG)
    * This is going to be a script that splits items in C or items in A into "atomised claims" (see literature below).
    * The SAFE architecture’s refined version of AFG (in long-form-factuality/third_party/factscore/atomic_facts.py), using “demons” that are dynamically swapped in and out of the system prompt (I believe) to ensure best context for the model doing the AFG seems the best of the bunch. We could customise the demons in the txt and json files in /demos.
    * So, each item like c1 is further split into c1-1, c1-2, c1-3 up to c1-k, at a variable k (some items can have more claims than others). Output would be a delimited string (ie a string of the claims delimited by like |). Another output for each item would be the k for that item -- that is, how many claims that the item has been split into. 
    * of course, this requires an LLM. the script should be able to take any LLM name on hugging face, so uses transformers library with templating for system/user prompts.
2. Abstention Detection
    * We need to ensure that, where a tested LLM provides answer a1 that doesn't actually answer the question, we detect this abstention and ensure that is recognised at the end of the evals. Abstentions are important to us - yes we want to know about misinformation, but we also want to know about where LLMs dodge the question. These should be two distinct scores coming out of our benchmark.
    * The way abstention detection is done in FActScore is simply not good enough. I don't have a solution for it yet. If you find something or figure something out go ahead and propose something for me.
3. Automatic fact verification (AFV)
    * this is the hardest part. for each prompt qi, we need to examine the atomised versions of ci and ai -- ie ci-1 up to ci-k, and ai-1 up to ai-k -- and figure out the accuracy of the evaluated model. So, the number of claims in ai that are present in the ground truth ci.
    * this is LLM-as-a-judge.
    * but what metric is used? I want to do some sort of accuracy@k, where we don't count accurate claims in ai beyond the number of claims in ci.  


I've put a short literature review below. Help me out!








Literature
Literature on the topic of benchmarking LLMs for factuality has almost entirely revolved around FActScore, a method introduced by [Min et al. (2023)](https://arxiv.org/abs/2305.14251) that provides a pipeline for either human or machine annotators to evaluate the truthfulness of LLM responses. The method first involves Atomic Fact Generation (AFG) asking an LLM a question – e.g. “Who is Thierry Henry?” – and splitting (“atomising”) the response into a set of “claims” (note, not sentences; a single sentence in an LLM response can have many supported or unsupported claims, so the atomisation needs to be more fine-grained). Afterward, each claim is checked for factuality (Atomised Fact Verification, AFV) against a suitable knowledge source (such as Thierry Henry’s Wikipedia article), and the percentage of claims that are supported by the knowledge source is reported as the resulting FActScore of the LLM for a prompt or set of prompts.

Min et al. utilise FActScore with human annotators conducting both AFG and AFV, but also release code to use a fully automated workflow with InstructGPT to generate an estimator of FActScore. The latter method has been utilised, adapted, and refined in subsequent literature.

For example, [Lage & Ostermann (2025)](https://arxiv.org/abs/2507.05965) introduce OpenFActScore that counters the original reliance on InstructGPT for AFG/AFV in the evaluation pipeline, forking the FActScore repository to facilitate the usage of any model on Hugging Face. Across numerous tests, they find some models better suited for AFG and others better suited for AFV, ultimately proposing a hybrid approach with Olmo and Gemma that has 99% Pearson correlation with InstructGPT (despite significantly lower scores, prompting future research).

Famously, the SAFE architecture ([Wei et al. 2024](https://arxiv.org/abs/2403.18802)) asks: rather than the knowledge base used for AFV being a constrained, static chunk of text, what if it was instead the entire internet? The architecture uses Google Search as a retrieval mechanism for AFV to provide massive amounts of evidence to dictate whether LLM responses are correct, outperforming human annotations and addressing some key limitations of FActScore. The paper also introduces an underlying metric for AFV – F1@K, which allows dynamic measurement of LLM factuality “up to the kth claim” for a hyperparameter k to ensure additional, perhaps unrelated facts reported by LLMs do not increase factuality measurement. Finally, the authors make tweaks across the FActScore codebase, including in AFG where they implement dynamic claim separation using “demons” to give LLMs more context. 

VERISCORE ([Song, Kim & Iyyer 2024](https://arxiv.org/abs/2406.19276)) notes potential downward biases in both FActScore and SAFE, where components of LLM responses (like “I’m sorry to hear that” or “I’m happy to help”) are potentially not filtered in AFG, are found in AFV, and, unsupported by a knowledge base, are marked as nonfactual and reduce LLM factuality scores. The authors expand AFG instructions to only generate “verifiable” claims. Similarly, OpenFActScore has an “abstention detection module” to detect where LLMs refuse to provide claims (see also [link](https://www.ibtimes.co.uk/chatgpt-admits-its-limit-new-rules-ban-specific-legal-health-money-tips-report-1751762)).
