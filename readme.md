# AI Agent Engineering Knowledge

This section contains answers for the AI Agent Engineering Knowledge questions

---

## 1. Describe differences between REST API, MCP in the context of AI.

- **REST API:**  
  Provides a stateless interface to interact with AI models. Applications send input and receive predictions in real time.

- **MCP (Model Control Plane):**  
  Manages the lifecycle of AI models, including deployment, versioning, monitoring, and scaling. Ensures models are reliable and maintainable.

*Summary:* REST API is for accessing AI models, MCP is for managing and orchestrating them.

---

## 2. How REST API, MCP, can improve the AI use case.

- **REST API:** Enables easy integration of AI into applications and services.  
- **MCP:** Ensures reliability, tracks performance, supports rollbacks, and manages multiple models efficiently.  

We can use both make AI deployment scalable, dependable, and maintainable.

---

## 3. How do you ensure that your AI agent answers correctly?

- Use high quality, clean, and representative data.  
- Validate models through testing and real world scenarios.  
- Monitor outputs for drift and errors.  
- Use explainability tools or fallback rules for uncertain predictions.  
- Continue retraining models based on feedback.

---

## 4. Docker / Containerized Environments

- Package code, dependencies, and environment consistently.  
- Ensure reproducibility across systems.  
- Enable scalable deployments and CI/CD integration.  
- Simplify version tracking and isolation of AI experiments.

---

## 5. Fine-Tuning LLMs from Raw

1. Select a pre-trained model.  
2. Collect and preprocess domain-specific data.  
3. Define training objectives and configure hyperparameters.  
4. Fine-tune using supervised learning or RLHF.  
5. Evaluate and iterate using validation datasets.  
6. Deploy via REST API or MCP with monitoring.

---

# Coding Test

This section contains answers for the Coding Test questions

---

## Insight customer-100000.csv 

**Rows:** 100,000  
**Columns:** `['Index', 'Customer Id', 'First Name', 'Last Name', 'Company', 'City', 'Country', 'Phone 1', 'Phone 2', 'Email', 'Subscription Date', 'Website']`

---

## Numeric Columns Summary

| Column | Count    | Mean      | Std        | Min | 25%       | 50%       | 75%       | Max     |
|--------|----------|-----------|------------|-----|-----------|-----------|-----------|---------|
| Index  | 100,000  | 50,000.5  | 28,867.66  | 1   | 25,000.75 | 50,000.5  | 75,000.25 | 100,000 |

---

## Top 5 Values per Categorical Column

**Customer Id:**  
- 53DBF8C8e33007b (1)  
- A2F27F7E8Ee7A7b (1)  
- 7E8EC8Bc5491Bbd (1)  
- cD90bD6fcF5E02C (1)  
- 62ca9D31c59dc7D (1)  

**First Name:**  
- Joan (183)  
- Audrey (182)  
- Bridget (182)  
- Anne (180)  
- Melinda (177)  

**Last Name:**  
- Campbell (139)  
- Carney (132)  
- Gardner (131)  
- Patterson (130)  
- Cisneros (127)  

**Company:**  
- Campbell Ltd (17)  
- Wilkerson Ltd (17)  
- Booker and Sons (16)  
- Acosta Ltd (16)  
- Mccarty and Sons (15)  

**City:**  
- Lake Frederick (16)  
- East Jeremy (15)  
- East Lee (15)  
- West Alec (15)  
- New Christopher (15)  

**Country:**  
- Congo (835)  
- Korea (820)  
- Saudi Arabia (463)  
- Pitcairn Islands (456)  
- Saint Martin (453)  

**Phone 1:**  
- 973-170-7283x8389 (1)  
- (948)810-1424x459 (1)  
- 077-245-2618 (1)  
- 001-450-998-6032 (1)  
- 559-467-0737x720 (1)  

**Phone 2:**  
- 001-270-479-8553x9053 (1)  
- +1-842-851-2545x429 (1)  
- (908)899-2270x156 (1)  
- 358-893-0736x3799 (1)  
- (509)631-4080x43607 (1)  

**Email:**  
- julia03@briggs.com (2)  
- kwalls@white.com (2)  
- ushields@saunders.com (2)  
- vgeorge@mendoza.com (2)  
- imitchell@church.com (2)  

**Subscription Date:**  
- 2020-12-11 (155)  
- 2020-05-22 (146)  
- 2020-01-18 (145)  
- 2022-02-13 (144)  
- 2020-10-31 (144)  

**Website:**  
- https://guzman.com/ (22)  
- http://www.maxwell.com/ (22)  
- https://www.nunez.com/ (22)  
- http://cooke.com/ (21)  
- https://boyle.com/ (21)

## Insight customers-2000000.csv

**Rows:** 2,000,000  
**Columns:** `['Index', 'Customer Id', 'First Name', 'Last Name', 'Company', 'City', 'Country', 'Phone 1', 'Phone 2', 'Email', 'Subscription Date', 'Website']`

---

## Numeric Columns Summary

| Column | Mean        | Variance           | Count     |
|--------|------------|------------------|-----------|
| Index  | 1,000,000.5 | 333,333,333,335.1 | 2,000,000 |

---

## Top 5 Values per Categorical Column

**Company:**  
- Bridges Group (152)  
- Conway LLC (148)  
- Cooke and Sons (147)  
- Shields Ltd (146)  
- Oneill Ltd (146)  

**Customer Id:**  
- 4962fdbE6Bfee6D (1)  
- 9b12Ae76fdBc9bE (1)  
- 39edFd2F60C85BC (1)  
- Fa42AE6a9aD39cE (1)  
- F5702Edae925F1D (1)  

**Email:**  
- meagan67@graham.com (3)  
- shardy@horne.com (2)  
- bradley65@haley.com (2)  
- ipatton@nolan.com (2)  
- jdonovan@shields.com (2)  

**Website:**  
- http://www.lloyd.com/ (256)  
- http://whitaker.com/ (255)  
- http://lewis.com/ (253)  
- https://www.contreras.com/ (244)  
- http://www.pacheco.com/ (242)  

**Phone 1:**  
- 9363752051 (2)  
- 001-078-920-4851 (2)  
- 540.584.0550 (2)  
- 779.345.5892 (2)  
- 1468463971 (2)  

**Phone 2:**  
- 473-464-5270 (2)  
- 702-550-9740 (2)  
- 9794059873 (2)  
- 493-421-2526 (2)  
- 920.905.8447 (2)  

**First Name:**  
- Mariah (3,082)  
- Samantha (3,040)  
- Jonathon (3,037)  
- Patricia (3,031)  
- Kerry (3,027)  

**Last Name:**  
- Bird (2,133)  
- Gill (2,133)  
- Hubbard (2,126)  
- Lara (2,119)  
- Lamb (2,115)  

**City:**  
- Bradleymouth (163)  
- Leemouth (156)  
- Ashleymouth (155)  
- Kirkmouth (153)  
- Barrymouth (151)  

**Country:**  
- Korea (16,240)  
- Congo (16,208)  
- Jordan (8,428)  
- Vietnam (8,388)  
- Suriname (8,375)  

**Subscription Date:**  
- 2022-02-27 (2,430)  
- 2020-07-07 (2,428)  
- 2020-06-16 (2,414)  
- 2021-06-04 (2,402)  
- 2021-07-04 (2,399)  

## Small vs Large Files Comparison
- Small files can be loaded fully into memory (e.g., with pandas) -> easy and fast for analysis.
- Large files exceed memory -> must be streamed row by row -> manual computation for stats and top values.
