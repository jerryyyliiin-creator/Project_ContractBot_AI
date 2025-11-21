CONTRACT_RISK_KNOWLEDGE_BASE = [
    # =================================================================
    # ==                     🚨 High-Risk Clauses                    ==
    # =================================================================
    {
        "risk_level": "High",
        "clause_type": "Unlimited or Unilateral Indemnification",
        "keywords": ["indemnify", "hold harmless", "defend", "damages", "losses", "liabilities", "claims", "costs", "expenses",
            "unlimited liability", "caused by", "resulting from", "arising out of", "negligence of the indemnified party"],
        "risk_analysis": "This is one of the most dangerous clauses in a contract, commonly referred to as a “blank check.” It requires our side to assume unlimited financial liability for the other party, even for their own mistakes or negligence. A single incident could result in catastrophic financial damage to the company. Such a clause must be strictly rejected or, at the very least, limited by a liability cap."
    },
    {
        "risk_level": "High",
        "clause_type": "Assignment of Core / Background IP",
        "keywords": ["intellectual property", "IP", "assigns", "transfers", "ownership", "vests in", "background IP",
            "pre-existing technology", "all rights, title, and interest", "work product"],
        "risk_analysis": "This clause is highly aggressive. It not only transfers the newly created intellectual property from the project but also attempts to deprive us of our core assets (background technology) that we already owned prior to the collaboration. Accepting this clause would be equivalent to giving away the company’s technological foundation for free, which would severely weaken our long-term competitiveness and future development."
    },
    {
        "risk_level": "High",
        "clause_type": "Unrestricted & Perpetual Data Rights",
        "keywords": ["data", "content", "perpetual", "irrevocable", "royalty-free", "sublicense", "distribute",
            "commercial purpose", "derivative works", "any purpose whatsoever"],
        "risk_analysis": "This clause grants the other party unlimited rights to permanently and freely exploit the data we provide, potentially even selling it or giving it to our competitors. In an era where data is an asset, this could trigger unpredictable commercial, legal, and reputational risks."
    },
    {
        "risk_level": "High",
        "clause_type": "Unilateral Termination for Convenience",
        "keywords": ["terminate", "for convenience", "at any time", "in its sole discretion", "without cause", "for any reason"],
        "risk_analysis": "A clause of extreme power imbalance. It allows the other party to withdraw at any time and without reason, even after we have invested significant time, money, and human resources. This would leave all our initial investments unrecoverable, making the partnership highly unstable and forcing us to bear the full risk of sunk project costs."
    },
    {
        "risk_level": "High",
        "clause_type": "Broad & Long-term Non-Compete / Exclusivity",
        "keywords": ["exclusive", "non-compete", "non-competition", "refrain from", "similar business", "affiliates"],
        "risk_analysis": "Overly broad exclusivity clauses (e.g., vague geographic scope, unclear business definitions, or inclusion of all affiliates) and excessively long durations (e.g., extending more than two years after contract termination) can completely lock the company’s development path. They prevent expansion into new markets or collaboration with more favorable partners, severely hindering the company’s core business growth."
    },
    {
        "risk_level": "High",
        "clause_type": "Uncapped Liquidated Damages",
        "keywords": ["liquidated damages", "penalty", "service credits", "uncapped", "without limitation"],
        "risk_analysis": "Although liquidated damages are common, if no reasonable overall cap is set (for example, not exceeding the total service fees of the previous 12 months), a single service disruption or delivery delay could result in compensation amounts far exceeding the actual value of the contract, creating disproportionately large financial risks."
    },

    # =================================================================
    # ==                    ⚠️ Medium-Risk Clauses                   ==
    # =================================================================
    {
        "risk_level": "Medium",
        "clause_type": "Extended Payment Terms",
        "keywords": ["payment", "net 60", "net 90", "net 120", "invoice", "payment terms", "days after receipt"],
        "risk_analysis": "Overly long payment terms (e.g., Net 60 or Net 90) can severely impact our cash flow health. Although the receivables will eventually be collected, during the waiting period we must bear financial pressure and may miss other investment opportunities, increasing both the opportunity cost of operations and financial risk."
    },
    {
        "risk_level": "Medium",
        "clause_type": "Vague Scope of Work (SOW) & Acceptance Criteria",
        "keywords": ["scope of work", "SOW", "acceptance criteria", "reasonable efforts", "as needed", "to the satisfaction of"],
        "risk_analysis": "Vaguely defined terms such as “all related” or “to their complete satisfaction” can be interpreted by the other party without limit, making the agreement highly susceptible to scope creep. This could force us to commit far more resources and manpower than anticipated, ultimately leading to project losses."
    },
    {
        "risk_level": "Medium",
        "clause_type": "Unequal or Low Limitation of Liability",
        "keywords": ["limitation of liability", "liability cap", "consequential damages", "indirect damages", "disclaim"],
        "risk_analysis": "A liability cap is meant to protect both parties, but if the cap is set too low (e.g., limited to just one month of service fees) or is unbalanced between the parties, we may be unable to obtain sufficient compensation in the event of the other party’s major breach. This creates a disproportionate relationship between risk and return."
    },
    {
        "risk_level": "Medium",
        "clause_type": "Overly Broad Non-Solicitation",
        "keywords": ["non-solicitation", "solicit", "poach", "employee", "contractor", "former employee", "indirectly"],
        "risk_analysis": "Overly broad non-solicitation clauses (e.g., covering all of the other party’s employees rather than only those involved in this project) could prevent us from hiring top talent in the open market, unreasonably restricting the company’s talent acquisition strategy."
    },
    {
        "risk_level": "Medium",
        "clause_type": "Automatic Renewal",
        "keywords": ["automatic renewal", "automatically renew", "term", "notice of non-renewal"],
        "risk_analysis": "This clause poses an operational risk. If, due to administrative oversight, we fail to issue a non-renewal notice within the specified timeframe, the company will be forced into renewal—even if the contract no longer aligns with current business interests or better alternatives exist—resulting in unnecessary financial burdens."
    },
    {
        "risk_level": "Medium",
        "clause_type": "Permissive Assignment Clause",
        "keywords": ["assignment", "assign", "transfer", "without prior written consent", "change of control"],
        "risk_analysis": "If a clause allows the other party to freely assign the contract to a third party without our consent (which could even be our competitor or a financially unstable company), it would create significant uncertainty and potential risks in the partnership."
    },

    # =================================================================
    # ==                       🔍 LOW-RISK CLAUSES                   ==
    # =================================================================
    {
        "risk_level": "Low", 
        "clause_type": "Standard Governing Law & Jurisdiction",
        "keywords": ["governing law", "jurisdiction", "venue", "applicable law", "choice of law"],
        "risk_analysis": "This is a standard clause with relatively low risk. However, it is important to ensure that the governing law and jurisdiction chosen are neutral and convenient for us. If the clause designates a distant or unfamiliar jurisdiction, the cost and difficulty of resolving future disputes will significantly increase."
    },
    {
        "risk_level": "Low",
        "clause_type": "Reasonable Confidentiality Period",
        "keywords": ["confidentiality", "non-disclosure", "NDA", "confidential information", "term"],
        "risk_analysis": "The risk is relatively low. A confidentiality period of three to five years for ordinary trade secrets is considered market standard. However, if the clause involves the company’s core technical know-how or permanent business secrets, it is advisable to negotiate for a longer protection period, or to require that the confidentiality obligation remain effective indefinitely."
    },
    {
        "risk_level": "Low",
        "clause_type": "Force Majeure",
        "keywords": ["force majeure", "act of god", "natural disaster", "pandemic", "epidemic", "cyber attack"],
        "risk_analysis": "The risk is relatively low, as this is a standard exemption clause. However, it is important to review whether the defined scope of events is fair and up to date. For example, in today’s business environment, it is critical for both parties to ensure that the clause explicitly covers non-traditional force majeure events such as pandemics, government actions, and large-scale cyberattacks."
    },
    {
        "risk_level": "Low",
        "clause_type": "Notices",
        "keywords": ["notices", "written notice", "email", "address", "contact person"],
        "risk_analysis": "The risk is minimal, as this is an administrative clause. However, when reviewing, it is essential to ensure that our designated mailing address, contact person, and email address are accurate and properly monitored to avoid missing any important legal notices (such as notices of breach or termination), which could otherwise result in the loss of rights."
    },
]

def get_risk_rubric_string():
    """Format the knowledge base into a clear and structured string to facilitate injection into the prompt."""
    rubric_parts = []
    for item in CONTRACT_RISK_KNOWLEDGE_BASE:
        part = (
            f"- **Risk Level: {item['risk_level']}**\n"
            f"  - **Clause Type**: {item['clause_type']}\n"
            f"  - **Keywords**: {', '.join(item.get('keywords', ['N/A']))}\n"
            f"  - **Business Risk Analysis**: {item['risk_analysis']}\n"
        )
        rubric_parts.append(part)
    return "\n".join(rubric_parts)