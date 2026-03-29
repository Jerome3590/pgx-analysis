### **README: Clinical OODA & AWS Analytics Architecture**

This repository contains the technical specifications and architectural diagrams for the study: **"Temporal Risk Modeling for Adverse Drug Events (ADEs) — A Machine Learning Process Mining Pipeline"**. The framework leverages Virginia's All Payer Claims Database (APCD) to identify higher-order prescribing patterns contributing to preventable hospitalizations.

## ---

**1\. Data Preprocessing Architecture (Figure 1\)**

The preprocessing engine is a modular 14-step pipeline designed for scalability and fault tolerance. It utilizes a **Bronze-Silver-Gold** S3 data lake pattern, with all transformations powered by **DuckDB** on a persistent **AWS EC2 x2iedn.8xlarge spot instance**.

* **Bronze Zone**: Houses raw APCD Medical and Pharmacy claims.

* **Silver Zone**: Contains normalized datasets and 14 distinct processing checkpoints for stateful recovery.

* **Gold Zone**: Stores curated cohorts (OPIOID\_ED, ADE, and Control) in partitioned Parquet format.

Code snippet

\\begin{tikzpicture}\[  
    node distance=1.1cm and 1.6cm,  
    base/.style={draw, thick, align=center, font=\\small\\sffamily, rounded corners, fill=white},  
    s3\_node/.style={base, draw=orange, fill=orange\!5, minimum width=2.2cm},  
    compute\_node/.style={base, draw=blue\!80\!black, fill=blue\!5, thick, minimum width=3.5cm},  
    arrow/.style={-{Stealth\[scale=1.2\]}, thick, draw=gray\!80}  
\]

    % Nodes  
    \\node\[base\] (sources) {APCD Claims\\\\(Pharmacy/Medical)};  
    \\node\[s3\_node, right=of sources\] (bronze) {\\iconS3 \\\\ \\textbf{Bronze Zone}};  
    \\node\[s3\_node, right=of bronze\] (silver) {\\iconS3 \\\\ \\textbf{Silver Zone}};  
    \\node\[s3\_node, right=of silver\] (gold) {\\iconS3 \\\\ \\textbf{Gold Zone}};

    \\node\[compute\_node, below=1.8cm of silver\] (ec2) {  
        \\iconEC2 \\quad \\iconDuckDB \\\\  
        \\textbf{DuckDB / x2iedn.8xlarge} \\\\  
        \\textit{Preprocessing Steps 1-14}  
    };

    \\node\[base, dashed, above=0.7cm of bronze\] (glue) {\\iconGlue \\ AWS Glue};

    % Connections  
    \\draw\[arrow\] (sources) \-- (bronze);  
    \\draw\[arrow\] (bronze) \-- (silver) node\[midway, above, font=\\tiny\] {Normalize};  
    \\draw\[arrow\] (silver) \-- (gold) node\[midway, above, font=\\tiny\] {Curate};  
      
    \\draw\[arrow, \<-\>\] (ec2.north) \-- (silver.south) node\[midway, left, font=\\tiny\] {Transforms};  
    \\draw\[arrow, \<-\>\] (ec2.north) \-- (gold.south);  
    \\draw\[arrow, dashed\] (glue) \-- (bronze);

    % Legend  
    \\node\[draw, gray, dotted, inner sep=0.4cm, fit=(bronze) (gold) (glue), label=above:AWS S3 Data Lake\] {};  
\\end{tikzpicture}

## ---

**2\. Interactive Insights & Multi-modal Analytics (Figure 2\)**

The analytical layer unifies unsupervised pattern discovery with supervised risk modeling. The platform operates via an interactive **Jupyter/Quarto** environment supporting dual kernels:

* **R (BupaR)**: Conducts temporal pathway analysis and calculates throughput times between drug administrations.

* **Python (CatBoost & FFA)**: Trains predictive models on frequent itemsets discovered via **FP-Growth** and applies **Formal Feature Attribution** for clinical interpretability.

Code snippet

\\begin{tikzpicture}\[  
    node distance=1.2cm and 2cm,  
    base/.style={draw, thick, align=center, font=\\small\\sffamily, rounded corners, fill=white},  
    gold\_node/.style={base, draw=yellow\!80\!black, fill=yellow\!10, minimum width=2.5cm},  
    kernel\_node/.style={base, draw=blue\!80\!black, fill=blue\!5, minimum width=3cm},  
    arrow/.style={-{Stealth\[scale=1.2\]}, thick, draw=gray\!80}  
\]

    % Nodes  
    \\node\[gold\_node\] (gold) {\\iconS3 \\\\ \\textbf{Gold Zone (S3)}};  
      
    \\node\[kernel\_node, right=of gold, yshift=1cm\] (r\_kernel) {  
        \\textbf{R Kernel} \\\\ BupaR Process Mining  
    };  
      
    \\node\[kernel\_node, right=of gold, yshift=-1cm\] (py\_kernel) {  
        \\textbf{Python Kernel} \\\\ CatBoost \+ FFA  
    };

    \\node\[base, right=2cm of r\_kernel, yshift=-1cm, minimum height=2cm\] (quarto) {  
        \\iconGithub \\\\ \\textbf{Interactive Insights} \\\\ (Jupyter / Quarto)  
    };

    % Connections  
    \\draw\[arrow\] (gold.east) \-- (r\_kernel.west);  
    \\draw\[arrow\] (gold.east) \-- (py\_kernel.west);  
    \\draw\[arrow\] (r\_kernel.east) \-- (quarto.north west);  
    \\draw\[arrow\] (py\_kernel.east) \-- (quarto.south west);  
      
    \\node\[draw, blue\!40, dashed, inner sep=0.5cm, fit=(r\_kernel) (py\_kernel), label=above:AI/ML Custom Analytics\] {};  
\\end{tikzpicture}

## ---

**3\. Risk Dashboard & Edge Delivery**

To ensure transparency and reproducibility, the final analytical artifacts and risk visualizations are delivered through a high-availability serverless stack.

* **Route 53 & CloudFront**: Provides low-latency access to the public-facing dashboard at jerome-dixon.io.

* **AWS Lambda & API Gateway**: Manages model artifact retrieval and user requests.  
* **AWS QuickSight**: Enables URL embedding of notebooks alongside native dashboards for collaborative decision support.

Code snippet

\\begin{tikzpicture}\[  
    node distance=1.1cm and 1.5cm,  
    base/.style={draw, thick, align=center, font=\\small\\sffamily, rounded corners, fill=white},  
    net\_node/.style={base, draw=purple, fill=purple\!5, minimum width=2.5cm},  
    arrow/.style={-{Stealth\[scale=1.2\]}, thick, draw=gray\!80}  
\]

    % Nodes  
    \\node\[base\] (artifacts) {\\iconS3 \\\\ Model Artifacts};  
    \\node\[net\_node, right=of artifacts\] (gateway) {\\iconGateway \\\\ API Gateway};  
    \\node\[net\_node, below=of gateway\] (lambda) {\\iconLambda \\\\ AWS Lambda};  
    \\node\[net\_node, right=of gateway\] (cf) {\\iconCF \\\\ CloudFront CDN};  
    \\node\[net\_node, right=of cf\] (route) {\\iconRoute \\\\ Route 53 \\\\ (jerome-dixon.io)};

    % Connections  
    \\draw\[arrow\] (artifacts) \-- (gateway);  
    \\draw\[arrow, \<-\>\] (gateway) \-- (lambda);  
    \\draw\[arrow\] (gateway) \-- (cf);  
    \\draw\[arrow\] (cf) \-- (route);

    % Boundary  
    \\node\[draw, purple\!40, dotted, inner sep=0.4cm, fit=(gateway) (cf) (route) (lambda), label=above:Risk Dashboard Delivery\] {};  
\\end{tikzpicture}

### ---

**Performance Highlights**

* **Full Run Runtime**: \~4 hours for all nine age cohorts.

* **Cost Efficiency**: Approximately $6.00 to $10.00 per full run using Spot Instances.

* **Scalability**: Processes millions of rows (8-15GB per base table) entirely in-memory.

Would you like me to generate the **Sankey diagram logic** for Figure 3 to visualize the high-frequency drug transitions identified in your ADE cohort?.

