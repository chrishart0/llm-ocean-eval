# V1 Big Five (OCEAN) Personality Evaluation for LLMs

## Overview
We’re proposing a V1 system to evaluate the Big Five personality traits—Openness (O), Conscientiousness (C), Extraversion (E), Agreeableness (A), and Neuroticism (N)—of major large language models (LLMs). The aim is to assess base personality profiles using a standardized test via API and share the results publicly. This sets a foundation for understanding LLM behavioral tendencies, with potential for later expansion (e.g., testing trait flexibility).

## Value Proposition
- **User Insight**: Enables users to choose LLMs based on personality “feel” (e.g., high E for outgoing, low N for calm).
- **Developer Feedback**: Highlights traits for refinement or branding purposes.
- **Research Base**: Provides a public dataset for LLM personality analysis.
- **Engagement**: Offers a fresh, relatable metric to generate interest (e.g., social media traction).

## V1 Scope
Focus on generating base Big Five scores for prominent LLMs using a consistent, repeatable approach. No exploration of trait divergence or advanced features—that’s for V2.

## Proposed Structure

### 1. Assessment Tool
- **Choice**: 44-item Big Five Inventory (BFI).
  - Covers O, C, E, A, N with statements scored 1-5 (disagree to agree).
  - Example: “I see myself as someone who is outgoing, sociable” (E).
- **Reason**: Widely recognized, used in prior LLM studies, and straightforward.

### 2. Approach
- **Method**: Query models with a neutral prompt to capture their default personality.
  - Example: “Rate how much you agree with this statement, answering as yourself, on a scale of 1 to 5: [BFI item].”
- **Reason**: Keeps results tied to inherent traits, avoids external influence.

### 3. Target Models
- **Candidates** (as of March 15, 2025):
  - GPT-4 (OpenAI)
  - Claude 3 (Anthropic)
  - Grok (xAI)
  - Llama 3 (Meta, if accessible)
  - Mixtral (Mistral)
- **Reason**: Represents leading models with broad use and varied designs.

### 4. Output
- **Format**: Scores per trait per model (e.g., O: 4.5, C: 3.5, E: 4.0, A: 3.8, N: 2.0).
- **Presentation**: Table for comparison.
  | Model    | Openness | Conscientiousness | Extraversion | Agreeableness | Neuroticism |
  |----------|----------|-------------------|--------------|---------------|-------------|
  | GPT-4    | 4.5      | 3.5               | 4.0          | 3.8           | 2.0         |
  | Claude 3 | 4.0      | 4.0               | 3.5          | 4.5           | 1.5         |
  | Grok     | 3.5      | 3.8               | 3.5          | 4.0           | 2.5         |
- **Details to Include**:
  - Date of evaluation.
  - Prompt wording.
  - Model versions tested.
  - Note on trial count (e.g., averaged results).
- **Reason**: Ensures clarity and reproducibility.

### 5. Sharing
- **Channels**: Publicly accessible platforms (e.g., GitHub, blog, social media).
- **Reason**: Maximizes reach and feedback.

## Future Considerations (V2)
- Measure trait divergence with varied prompts (e.g., “Act shy” vs. “Be bold”).
- Track changes over time or model updates.
- Add interactive elements for broader appeal.

## Challenges
- Ensuring prompt neutrality to avoid skewing results.
- Handling model access limitations.
- Accounting for score variability across runs.

## Next Steps
Engineers: Please assess feasibility, propose an execution plan, and identify any resource needs. Consider whether a basic divergence test could fit into V1. Goal is a quick, functional MVP to validate the concept.