# Multi-Phase Fine-tuned Gemma2 Models for Enhanced Multilingual Capabilities

[![Models]([https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow](https://www.kaggle.com/models/mahdiseddigh/gemma2))]

This repository contains a family of Gemma2 language models that have undergone a comprehensive multi-phase fine-tuning process to significantly enhance their multilingual abilities and task-specific performance. We leverage Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically Low-Rank Adaptation (LoRA), to adapt the base Gemma2 models efficiently.

## Project Overview

This project aims to develop robust multilingual language models by systematically fine-tuning the Gemma2 architecture through several distinct phases. Each phase builds upon the previous one, progressively enhancing the model's capabilities:

*   **Phase 1: Base Multilingual Fine-tuning:** This initial phase focuses on adapting the pre-trained Gemma2 model to perform well on a broad range of languages. We fine-tune the model on a large-scale multilingual dataset to improve its general linguistic understanding across different languages.

*   **Phase 2: Multilingual Instruction Fine-tuning:**  Building upon the base multilingual model, this phase aims to align the model with user instructions in various languages. We fine-tune the model on a multilingual instruction dataset, enabling it to better understand and respond to prompts and commands in different linguistic contexts.

*   **Phase 3: Direct Preference Optimization (DPO) Fine-tuning:** This phase focuses on refining the model's output quality and aligning it with human preferences. Using a specially crafted multilingual Direct Preference Optimization (DPO) dataset, we train the model to generate outputs that are considered more helpful, harmless, and honest.

*   **Phase 4: Domain-Specific Fine-tuning (Books):**  In this final phase (optional and ongoing), we specialize the models further by fine-tuning them on a diverse collection of books. This phase aims to enhance the model's ability to understand and generate longer-form text, improve its creative writing skills, and potentially adapt it to specific literary styles.

## Model Family

This repository hosts the following fine-tuned Gemma2 models:

*   **`gemma2-[9b]-base-multilingual`**: The models resulting from Phase 1. These models demonstrate improved general multilingual text generation capabilities compared to the base Gemma2 model.

*   **`gemma2-[9b]-multilingual-instruct`**: The models resulting from Phase 2. These models are optimized for following instructions in multiple languages and are recommended for general multilingual applications.
*   **`gemma2-[2b]-multilingual-dpo`**: The model resulting from Phase 3. This model has been fine-tuned using Direct Preference Optimization to generate higher-quality and more preferred outputs.

*   **`gemma2-[2b]-book-fine-tuned`**:  The models resulting from Phase 4. These models are specialized for book-related content and may exhibit enhanced capabilities in narrative generation and understanding.

**Note:**  Model availability may vary. Check the repository contents for the specific models currently available.

## 🚀 Usage

You can easily utilize these fine-tuned Gemma2 models for various multilingual text generation tasks.

**Loading the Models:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "path/gemma2-2b-multilingual-instruct" # Replace with the desired model name

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Generating Text:**

```python
prompt = "Translate 'Hello, world!' to Spanish."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Important Considerations:**

*   **Tokenizer:** Ensure you use the correct tokenizer associated with the specific Gemma2 model you are loading. The `AutoTokenizer.from_pretrained()` method will automatically fetch the appropriate tokenizer.
*   **Multilingual Applications:** The `-multilingual-instruct` models are generally recommended for a wide range of multilingual tasks due to their instruction-following capabilities.

## 🛠️ Implementation Details

The models were fine-tuned using the following infrastructure and techniques:

*   **Fine-tuning Library:**  The primary library used for fine-tuning was `transformers` along with the `trl` (Transformer Reinforcement Learning) library for DPO.
*   **Parameter-Efficient Fine-tuning (PEFT):** Low-Rank Adaptation (LoRA) was employed to efficiently fine-tune the large Gemma2 models while minimizing computational costs and memory requirements.
*   **Hardware:**
    *   **Phase 1 & 2:** Training for the 2B and 9B parameter models was conducted on Kaggle Notebooks, leveraging 2x Nvidia T4 GPUs and TPUv3.
    *   **Phase 3:** The DPO fine-tuning was performed on a colab Notebook equipped with a single Nvidia A100 40GB GPU.
    *   **Phase 4:** This phase used Nvidia P100 GPUs.
*   **Quantization:**  Quantization techniques (e.g., bitsandbytes) were utilized during training to further reduce memory footprint and potentially accelerate computations.

## 📚 Data Overview

The models were trained on carefully selected datasets for each phase:

*   **Phase 1 (Base Multilingual Fine-tuning):** The **C4 multilingual dataset** ([Hugging Face Dataset Link](https://huggingface.co/datasets/allenai/c4)) provided a massive corpus of text in numerous languages, enabling the model to develop a strong foundation in multilingual understanding.

*   **Phase 2 (Multilingual Instruction Fine-tuning):** The **Aya multilingual instruct dataset** ([Hugging Face Dataset Link](https://huggingface.co/datasets/CohereForAI/aya_dataset)) was used to train the models to follow instructions effectively across different languages. This dataset contains a diverse set of instruction-following examples in multiple languages.

*   **Phase 3 (DPO Fine-tuning):** A **synthetic DPO dataset** was created based on the Aya dataset. This dataset consists of carefully curated preference pairs, indicating which model output is preferred over another for a given instruction. This allows the model to learn to generate more desirable responses.

*   **Phase 4 (Book Fine-tuning):**  A dataset comprising **various books** was utilized for this phase. ["This dataset includes a mix of fiction and non-fiction books from sources like Project Gutenberg and the Open Library."].

## ➡️ Future Directions

This project is continuously evolving. Potential future enhancements include:

*   **Expanding Language Coverage:**  Further fine-tuning on datasets with even more diverse and low-resource languages.
*   **Improving DPO Dataset Quality:** Refining the DPO dataset with more nuanced and high-quality preference data.
*   **Exploring Different PEFT Techniques:** Investigating other parameter-efficient fine-tuning methods beyond LoRA.
*   **Community Contributions:** Encouraging contributions from the community in the form of new datasets, fine-tuning scripts, and model evaluations.
*   **Rigorous Evaluation:** Implementing comprehensive evaluation benchmarks to assess the performance of the models across various multilingual tasks.

## 🤝 Contributing

Contributions to this project are highly welcome! If you have ideas for improvements, new datasets, or have identified issues, please feel free to:

*   **Submit pull requests** with your proposed changes.
*   **Open issues** to report bugs or suggest new features.

## 📜 License

This project is licensed under the [Apache 2.0 License]([LICENSE](https://github.com/Mhdaw/Gemma2/blob/main/LICENSE)).

## 🙏 Acknowledgements

We gratefully acknowledge the following resources and libraries that made this project possible:

*   **Google for the Gemma2 model family:**  Providing the foundational models for this work.
*   **Hugging Face `transformers`, `datasets`, `trl`, and `peft` libraries:**  Essential tools for working with and fine-tuning large language models.
*   **The creators of the C4 and Aya datasets:**  For providing valuable multilingual data for training.
*   **Kaggle:** For providing the computational resources necessary for fine-tuning these large models.
*   **Colab:**  For providing the computational resources necessary for fine-tuning these large models.
