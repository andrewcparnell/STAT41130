# STAT41130 – Artificial Intelligence for Weather and Climate

**Author:** Andrew Parnell  
**Institution:** University College Dublin  
**Inspired by:** [AI by Hand – Tom Yeh](https://www.byhand.ai)  
**Official GitHub:** [https://github.com/andrewcparnell/STAT41130](https://github.com/andrewcparnell/STAT41130)

---

## 🧭 Overview

**STAT41130: AI for Weather and Climate** is a hands-on course designed to introduce students to machine learning methods for weather and climate applications.  
It combines manual “AI by Hand” exercises, Python coding, and real-world datasets to build understanding from basic linear regression through to modern neural networks and the ECMWF Anemoi framework.

The course bridges meteorology and artificial intelligence — ideal for students from both backgrounds who want to explore AI methods for forecasting, modelling, and data-driven science.

---

## 🎯 Aims

- Understand how neural networks extend traditional linear regression.  
- Learn key concepts of forward and backward propagation.  
- Implement and train models using **PyTorch**.  
- Explore deep learning architectures: CNNs, RNNs, Transformers, and GNNs.  
- Apply AI models to weather and climate data using the **Anemoi** framework from ECMWF.

---

## 🧩 Course Structure

**Format:**  
- 4 + 4 days intensive format  
- Morning: ~2 hours of lectures  
- Midday: guided coding session  
- Afternoon: 2–3 hour group projects and presentations  

**Content progression:**
1. **Linear Regression & Neural Networks**
2. **Deep Learning Fundamentals**
3. **Convolutional Neural Networks**
4. **Recurrent Neural Networks**
5. **Transformers**
6. **Graph Neural Networks**
7. **Probabilistic Forecasting**
8. **Anemoi Ecosystem: Graphs, Models, and Training**

---

## 🧱 Repository Contents

| Folder | Description |
|--------|--------------|
| `/slides` | Lecture slides in PowerPoint format (e.g., `D1C1_LR_NNs.pptx`) |
| `/code` | Python scripts and Jupyter notebooks for coding labs |
| `/worksheets` | PDF worksheets adapted from **AI by Hand** by Tom Yeh |
| `/setup` | Installation instructions and requirements files for Linux and Windows |
| `/data` | Example datasets for exercises (ERA5, Anemoi samples, etc.) |

---

## 🧠 Included Worksheets

The **AI by Hand** workbooks provide exercises to understand neural networks through manual calculation before coding

---

## ⚙️ Setup Instructions

### Recommended Environment
- **OS:** Ubuntu (preferred) or Windows  
- **Editor:** Visual Studio Code  
- **Python:** 3.11 

### Installation
Clone the repository:
```bash
git clone https://github.com/andrewcparnell/STAT41130.git
cd STAT41130
```

Install dependencies:
```bash
# For Linux
pip install -r requirements_ECMWF.txt

# For Windows
pip install -r requirements_ECMWF_win.txt
```

For GPU support, ensure you have a CUDA-compatible version of PyTorch as per [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🌦️ The Anemoi Framework

This course uses ECMWF’s **Anemoi** system — a modular ecosystem for machine learning in weather forecasting.

- **anemoi-graphs** – define graph structures for models  
- **anemoi-models** – provides neural network architectures (GNNs, Transformers, etc.)  
- **anemoi-training** – handles data loading, model training, and distributed computing  

Students will explore these packages using simple examples before scaling to larger datasets.

References:  
- [Anemoi Graphs Documentation](https://anemoi.readthedocs.io/projects/graphs/en/latest/)  
- [Anemoi Models Documentation](https://anemoi.readthedocs.io/projects/models/en/latest/)  
- [Anemoi Training Documentation](https://anemoi.readthedocs.io/projects/training/en/latest/)

---

## 💡 Learning Philosophy

This course emphasises:
- **Understanding by doing** – every lecture links to coding or hand exercises.  
- **Bridging theory and practice** – from matrix operations to full neural networks.  
- **Interdisciplinary collaboration** – between AI and meteorology students.  
- **Open-source tools** – to encourage exploration beyond the classroom.  

---

## 🧑‍💻 Contributing

If you find errors or have improvements:
1. Open an issue in this repository (intermediate)
2. Create a pull request with a fix (advanced and most helpful)
3. Or simply let the instructor know (basic)

---

## 📜 License

Course materials © 2025 Andrew Parnell, University College Dublin.  
Anemoi packages © ECMWF, licensed under Apache 2.0.
