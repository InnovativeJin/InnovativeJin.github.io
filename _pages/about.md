---
layout: about
title: about
permalink: /
subtitle: M.S. Candidate · <a href='https://cvlab.yonsei.ac.kr'>Computer Vision Lab, Yonsei University</a> · <a href='https://vig.kist.re.kr'>Visual Intelligence Group, KIST</a>

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p>Visual Intelligence Group</p>
    <p>Korea Institute of Science and Technology</p>
    <p>Seoul, Republic of Korea</p>

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

I am an M.S. candidate in the **YU–KIST cooperative graduate program**, jointly advised by **Dr.&nbsp;Heeseung Choi** at the [Visual Intelligence Group, KIST](https://vig.kist.re.kr) and **Prof.&nbsp;Bumsub Ham** at the [Computer Vision Lab, Yonsei University](https://cvlab.yonsei.ac.kr). My research is conducted at KIST.

<h2>Research Interests</h2>

<div style="margin:1rem 0 1.25rem; padding:1rem 1.25rem; border-left:3px solid var(--global-theme-color); background:var(--global-card-bg-color); border-radius:0 6px 6px 0;">
  <div style="font-family:'JetBrains Mono',ui-monospace,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--global-theme-color); margin-bottom:0.5rem;">Multimodal Learning for Predictive Physical Interaction Understanding</div>
  I develop <strong>multimodal models</strong> that understand how agents interact with objects and reason about the <strong>physical consequences of their actions</strong>. Building on my work in temporal action analysis, vision–language retrieval, and domain generalization, I aim to learn transferable representations of motion, object states, and interaction context from multimodal observations — with the ultimate goal of <strong>anticipating future interactions</strong> and action outcomes to support reliable decision-making in dynamic, real-world environments.
</div>

<div class="ed-keywords">
  <div class="ed-keywords-label">Research keywords</div>
  <div class="ed-keywords-chips">
    <span>Multimodal Learning</span>
    <span>Vision-Language Models</span>
    <span>Temporal Action Understanding</span>
    <span>Person Re-Identification</span>
    <span>Domain Generalization</span>
  </div>
</div>

<h2>What I've Worked On</h2>

- **Multimodal Human Proficiency Estimation (SkillLens)** — Action-guided Prototype Dynamics Margin (PDM) and CORAL-based ordinal calibration for proficiency estimation; **54.2%** accuracy on Ego-Exo4D (+4.1pp over the strongest evaluated baseline). *CVPR 2026 Workshop on MOMA (Oral).* [[project page](https://skill-lens.net/)]
- **Robust Text-to-Image Person Re-Identification** *(ongoing, KIST)* — Text-based person retrieval for missing-person and suspect searches where only natural-language descriptions are available.
- **Federated Domain Generalization for Person Re-Identification** *(KIST)* — Identity-ranking diagnostics and leave-one-client-out analysis for a deployed CCTV missing-person search system scaling across heterogeneous municipal environments.
- **Pose-Robust Representation Learning for Person Re-Identification (SwapMixer)** *(KIST)* — Generation-free feature-level augmentation combining identity representations with pose embeddings from different identities.
- **Knowledge Distillation for Open-Vocabulary Object Detection** *(Yonsei B.S. capstone)* — Similarity-preserving distillation aligning teacher–student relational representations; improved novel-class box AP50 on COCO by **+1.2** points (23.1 → 24.3).

See the [publications]({{ '/publications/' | relative_url }}) and [projects]({{ '/projects/' | relative_url }}) pages for the full list.
