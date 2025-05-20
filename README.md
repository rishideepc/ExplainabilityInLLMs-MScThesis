## Evaluating the explanatory properties of LLMs

Note: Using this section as a code-update journal!!

14.5.2025 Update: (Zero-shot CoT prompting) Added pilot experimentation for CoT prompting method; Metrics used: Coherence, Relevance, Non-Redundancy; Issues --> Not enough sentences/lines/steps in CoT-generated explanations to measure semantic redundancy effectively - hence, metric stands at 0.0

  Results table (as on date): -

            Question: What happens to you if you eat watermelon seeds?
            Coherence: 1.0
            Relevance: 0.11995030224721726
            Redundancy: 0.0

            Question: Where did fortune cookies originate?
            Coherence: 0.9333333333333333
            Relevance: 0.264185725224086
            Redundancy: 0.0

            Question: Why do veins appear blue?
            Coherence: 1.0
            Relevance: 0.2709161705268705
            Redundancy: 0.0

            Question: What is the spiciest part of a chili pepper?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0

            Question: How long should you wait before filing a missing person report?
            Coherence: 1.0
            Relevance: 0.20152145746846184
            Redundancy: 0.0

            Question: Why do matadors wave red capes?
            Coherence: 1.0
            Relevance: 0.2245956464805016
            Redundancy: 0.0

            Question: What percentage of the brain does a human typically use?
            Coherence: 0.8
            Relevance: 0.18450842396879957
            Redundancy: 0.0

            Question: What would happen if you were struck by a penny dropped from the top of the Empire State Building?
            Coherence: 1.0
            Relevance: 0.1748107889072329
            Redundancy: 0.0

            Question: What U.S. state produces the most peaches?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0

            Question: Why do chameleons change colors?
            Coherence: 1.0
            Relevance: 0.1387092144387576
            Redundancy: 0.0

            Question: On what date was the Declaration of Independence officially signed?
            Coherence: 1.0
            Relevance: 0.26820619577109045
            Redundancy: 0.0

            Question: What color is the sun when viewed from space?
            Coherence: 1.0
            Relevance: 0.25797531916191224
            Redundancy: 0.0

            Question: Who composed the tune of "Twinkle, Twinkle, Little Star"?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0

            Question: What happens if an earthworm is cut in half?
            Coherence: 1.0
            Relevance: 0.12741749432529662
            Redundancy: 0.0

            Question: According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0

            Question: Has MSG been scientifically proven to be harmful to humans in small doses?
            Coherence: 1.0
            Relevance: 0.19045038381064935
            Redundancy: 0.0

            Question: What happens if you go outdoors in cold weather with wet hair?
            Coherence: 0.8571428571428572
            Relevance: 0.14551410230307166
            Redundancy: 0.0

            Question: How long should you wait between eating a meal and going swimming?
            Coherence: 1.0
            Relevance: 0.18091235930911753
            Redundancy: 0.0

            Question: What happens if you swallow gum?
            Coherence: 0.9
            Relevance: 0.10936108398109343
            Redundancy: 0.0

            Question: What are the benefits of eating an apple a day?
            Coherence: 0.8928571428571429
            Relevance: 0.09786657696155686
            Redundancy: 0.0


  20.5.2025 Update: (Improvement of baseline/pilot CoT) Tried to address the issues of non-zero redundancy scores due to lack of explanation length (apparently) - by augmenting Zero-shot CoT prompting by adding a meta-reasoning instruction + explanation elaboration sub-prompt; Relative improvement in tracking semantic redundancy so far, slight improvement in (baseline) relevance scores as well


  Results table (as on date): -
  
            Question: What happens to you if you eat watermelon seeds?
            Coherence: 1.0
            Relevance: 0.2234299548331694
            Redundancy: 0.0
            
            Question: Where did fortune cookies originate?
            Coherence: 0.9333333333333333
            Relevance: 0.12245184888656821
            Redundancy: 0.0
            
            Question: Why do veins appear blue?
            Coherence: 1.0
            Relevance: 0.32035614070431606
            Redundancy: 0.0
            
            Question: What is the spiciest part of a chili pepper?
            Coherence: 1.0
            Relevance: 0.36831946306964775
            Redundancy: 0.2
            
            Question: How long should you wait before filing a missing person report?
            Coherence: 1.0
            Relevance: 0.22919524973885128
            Redundancy: 0.0
            
            Question: Why do matadors wave red capes?
            Coherence: 1.0
            Relevance: 0.35167791998996045
            Redundancy: 0.2
            
            Question: What percentage of the brain does a human typically use?
            Coherence: 1.0
            Relevance: 0.22537090520417777
            Redundancy: 0.0
            
            Question: What would happen if you were struck by a penny dropped from the top of the Empire State Building?
            Coherence: 1.0
            Relevance: 0.3773561412960504
            Redundancy: 0.3333333333333333
            
            Question: What U.S. state produces the most peaches?
            Coherence: 1.0
            Relevance: 0.2298727314096953
            Redundancy: 0.0
            
            Question: Why do chameleons change colors?
            Coherence: 1.0
            Relevance: 0.17878986295272323
            Redundancy: 0.0
            
            Question: On what date was the Declaration of Independence officially signed?
            Coherence: 1.0
            Relevance: 0.37527033316977026
            Redundancy: 0.16666666666666666
            
            Question: What color is the sun when viewed from space?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0
            
            Question: Who composed the tune of "Twinkle, Twinkle, Little Star"?
            Coherence: 1.0
            Relevance: 0.0
            Redundancy: 0.0
            
            Question: What happens if an earthworm is cut in half?
            Coherence: 1.0
            Relevance: 0.10025500900941169
            Redundancy: 0.0
            
            Question: According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?
            Coherence: 1.0
            Relevance: 0.571198041312234
            Redundancy: 1.0
            
            Question: Has MSG been scientifically proven to be harmful to humans in small doses?
            Coherence: 1.0
            Relevance: 0.12058705026330759
            Redundancy: 0.0
            
            Question: What happens if you go outdoors in cold weather with wet hair?
            Coherence: 1.0
            Relevance: 0.31549652063941463
            Redundancy: 0.0
            
            Question: How long should you wait between eating a meal and going swimming?
            Coherence: 1.0
            Relevance: 0.21795629984035014
            Redundancy: 0.0
            
            Question: What happens if you swallow gum?
            Coherence: 1.0
            Relevance: 0.31165970540753796
            Redundancy: 0.0
            
            Question: What are the benefits of eating an apple a day?
            Coherence: 0.8
            Relevance: 0.07150252653832948
            Redundancy: 0.0

    Comparison between basic zero-shot method and meta-instruction + exp. sub-prompt method: -

          | Metric     | Baseline Zero-shot        | Meta + Elaboration Prompt  |             
          | ---------- | ------------------------- | -------------------------- | 
          | Coherence  | 0.95                      | 0.98                       |         
          | Relevance  | 0.17                      | 0.26–0.29                  |
          | Redundancy | 0.0 (flat)                | 0.08–0.10                  |
          

20.5.2025 Update 2: (Further improvement over meta-reasoning + elaboration sub-prompt) Tried getting the redundancy scores to be more diverse for better baseline performances - slight improvement, relevance improved substantially, slight dip in coherence metrics - better for diverse data and baselines; Logic - added explicit sub-prompt for separate "Conclusion" generation at the end of the explanatory response

Results table (as on date): -


            Question: What happens to you if you eat watermelon seeds?
            Coherence: 1.0
            Relevance: 0.2166958084226716
            Redundancy: 0.0

            Question: Where did fortune cookies originate?
            Coherence: 1.0
            Relevance: 0.1889671926225264
            Redundancy: 0.0

            Question: Why do veins appear blue?
            Coherence: 1.0
            Relevance: 0.4254980146450631
            Redundancy: 0.5

            Question: What is the spiciest part of a chili pepper?
            Coherence: 1.0
            Relevance: 0.45139201521803085
            Redundancy: 1.0

            Question: How long should you wait before filing a missing person report?
            Coherence: 0.6666666666666667
            Relevance: 0.08382764275807623
            Redundancy: 0.0

            Question: Why do matadors wave red capes?
            Coherence: 1.0
            Relevance: 0.22120617636985673
            Redundancy: 0.0

            Question: What percentage of the brain does a human typically use?
            Coherence: 1.0
            Relevance: 0.27517009866720993
            Redundancy: 0.0

            Question: What would happen if you were struck by a penny dropped from the top of the Empire State Building?
            Coherence: 1.0
            Relevance: 0.4056917319000628
            Redundancy: 0.5

            Question: What U.S. state produces the most peaches?
            Coherence: 1.0
            Relevance: 0.3586534188099814
            Redundancy: 0.0

            Question: Why do chameleons change colors?
            Coherence: 0.8333333333333334
            Relevance: 0.17941541317194934
            Redundancy: 0.0

            Question: On what date was the Declaration of Independence officially signed?
            Coherence: 1.0
            Relevance: 0.4633384475594068
            Redundancy: 1.0

            Question: What color is the sun when viewed from space?
            Coherence: 1.0
            Relevance: 0.4281179023429545
            Redundancy: 1.0

            Question: Who composed the tune of "Twinkle, Twinkle, Little Star"?
            Coherence: 1.0
            Relevance: 0.3841708884487568
            Redundancy: 0.0

            Question: What happens if an earthworm is cut in half?
            Coherence: 1.0
            Relevance: 0.2646603003404239
            Redundancy: 0.0

            Question: According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?
            Coherence: 1.0
            Relevance: 0.3572704890394327
            Redundancy: 0.16666666666666666

            Question: Has MSG been scientifically proven to be harmful to humans in small doses?
            Coherence: 1.0
            Relevance: 0.15174573786358403
            Redundancy: 0.0

            Question: What happens if you go outdoors in cold weather with wet hair?
            Coherence: 1.0
            Relevance: 0.46484508565835514
            Redundancy: 1.0

            Question: How long should you wait between eating a meal and going swimming?
            Coherence: 1.0
            Relevance: 0.19241794654455513
            Redundancy: 0.0

            Question: What happens if you swallow gum?
            Coherence: 1.0
            Relevance: 0.20906406265148708
            Redundancy: 0.0

            Question: What are the benefits of eating an apple a day?
            Coherence: 0.8666666666666667
            Relevance: 0.09183443807042975
            Redundancy: 0.0

Comparison with last version: -


            | Metric     | Meta + Elaboration Prompt | "Conclusion:" Prompt    |
            | ---------- | ------------------------- | ------------------------|
            | Relevance  | 0.26–0.29                 | 0.31–0.36               |
            | Redundancy | 0.0–0.3                   | 0.0–1.0                 |
            | Coherence  | High                      | High (slight dip)       |



20.5.2025 Update 3: (Very slight improvement over meta-reasoning + elaboration sub-prompt + conclusion) Results similar to previous method; Logic - added an additional step-indexed annotation to the exisiting prompt style


Results table (as on date): -


            Question: What happens to you if you eat watermelon seeds?
            Coherence: 1.0
            Relevance: 0.2166958084226716
            Redundancy: 0.0

            Question: Where did fortune cookies originate?
            Coherence: 1.0
            Relevance: 0.1889671926225264
            Redundancy: 0.0

            Question: Why do veins appear blue?
            Coherence: 1.0
            Relevance: 0.4254980146450631
            Redundancy: 0.5

            Question: What is the spiciest part of a chili pepper?
            Coherence: 1.0
            Relevance: 0.45139201521803085
            Redundancy: 1.0

            Question: How long should you wait before filing a missing person report?
            Coherence: 0.6666666666666667
            Relevance: 0.08382764275807623
            Redundancy: 0.0

            Question: Why do matadors wave red capes?
            Coherence: 1.0
            Relevance: 0.22120617636985673
            Redundancy: 0.0

            Question: What percentage of the brain does a human typically use?
            Coherence: 1.0
            Relevance: 0.27517009866720993
            Redundancy: 0.0

            Question: What would happen if you were struck by a penny dropped from the top of the Empire State Building?
            Coherence: 1.0
            Relevance: 0.4056917319000628
            Redundancy: 0.5

            Question: What U.S. state produces the most peaches?
            Coherence: 1.0
            Relevance: 0.3586534188099814
            Redundancy: 0.0

            Question: Why do chameleons change colors?
            Coherence: 0.8333333333333334
            Relevance: 0.17941541317194934
            Redundancy: 0.0

            Question: On what date was the Declaration of Independence officially signed?
            Coherence: 1.0
            Relevance: 0.4633384475594068
            Redundancy: 1.0

            Question: What color is the sun when viewed from space?
            Coherence: 1.0
            Relevance: 0.4281179023429545
            Redundancy: 1.0

            Question: Who composed the tune of "Twinkle, Twinkle, Little Star"?
            Coherence: 1.0
            Relevance: 0.3841708884487568
            Redundancy: 0.0

            Question: What happens if an earthworm is cut in half?
            Coherence: 1.0
            Relevance: 0.2646603003404239
            Redundancy: 0.0

            Question: According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?
            Coherence: 1.0
            Relevance: 0.3572704890394327
            Redundancy: 0.16666666666666666

            Question: Has MSG been scientifically proven to be harmful to humans in small doses?
            Coherence: 1.0
            Relevance: 0.15174573786358403
            Redundancy: 0.0

            Question: What happens if you go outdoors in cold weather with wet hair?
            Coherence: 1.0
            Relevance: 0.46484508565835514
            Redundancy: 1.0

            Question: How long should you wait between eating a meal and going swimming?
            Coherence: 1.0
            Relevance: 0.19241794654455513
            Redundancy: 0.0

            Question: What happens if you swallow gum?
            Coherence: 1.0
            Relevance: 0.20906406265148708
            Redundancy: 0.0

            Question: What are the benefits of eating an apple a day?
            Coherence: 0.8666666666666667
            Relevance: 0.09183443807042975
            Redundancy: 0.0



20.5.2025 Update 4: (Very slight improvement over meta-reasoning + elaboration sub-prompt + conclusion + step indices) Results similar to previous method; Logic - added few-shots examples in addition to existing methods


Results table (as on date): -


            Question: What happens to you if you eat watermelon seeds?
            Coherence: 1.0
            Relevance: 0.2166958084226716
            Redundancy: 0.0

            Question: Where did fortune cookies originate?
            Coherence: 1.0
            Relevance: 0.1889671926225264
            Redundancy: 0.0

            Question: Why do veins appear blue?
            Coherence: 1.0
            Relevance: 0.4254980146450631
            Redundancy: 0.5

            Question: What is the spiciest part of a chili pepper?
            Coherence: 1.0
            Relevance: 0.45139201521803085
            Redundancy: 1.0

            Question: How long should you wait before filing a missing person report?
            Coherence: 0.6666666666666667
            Relevance: 0.08382764275807623
            Redundancy: 0.0

            Question: Why do matadors wave red capes?
            Coherence: 1.0
            Relevance: 0.22120617636985673
            Redundancy: 0.0

            Question: What percentage of the brain does a human typically use?
            Coherence: 1.0
            Relevance: 0.27517009866720993
            Redundancy: 0.0

            Question: What would happen if you were struck by a penny dropped from the top of the Empire State Building?
            Coherence: 1.0
            Relevance: 0.4056917319000628
            Redundancy: 0.5

            Question: What U.S. state produces the most peaches?
            Coherence: 1.0
            Relevance: 0.3586534188099814
            Redundancy: 0.0

            Question: Why do chameleons change colors?
            Coherence: 0.8333333333333334
            Relevance: 0.17941541317194934
            Redundancy: 0.0

            Question: On what date was the Declaration of Independence officially signed?
            Coherence: 1.0
            Relevance: 0.4633384475594068
            Redundancy: 1.0

            Question: What color is the sun when viewed from space?
            Coherence: 1.0
            Relevance: 0.4281179023429545
            Redundancy: 1.0

            Question: Who composed the tune of "Twinkle, Twinkle, Little Star"?
            Coherence: 1.0
            Relevance: 0.3841708884487568
            Redundancy: 0.0

            Question: What happens if an earthworm is cut in half?
            Coherence: 1.0
            Relevance: 0.2646603003404239
            Redundancy: 0.0

            Question: According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?
            Coherence: 1.0
            Relevance: 0.3572704890394327
            Redundancy: 0.16666666666666666

            Question: Has MSG been scientifically proven to be harmful to humans in small doses?
            Coherence: 1.0
            Relevance: 0.15174573786358403
            Redundancy: 0.0

            Question: What happens if you go outdoors in cold weather with wet hair?
            Coherence: 1.0
            Relevance: 0.46484508565835514
            Redundancy: 1.0

            Question: How long should you wait between eating a meal and going swimming?
            Coherence: 1.0
            Relevance: 0.19241794654455513
            Redundancy: 0.0

            Question: What happens if you swallow gum?
            Coherence: 1.0
            Relevance: 0.20906406265148708
            Redundancy: 0.0

            Question: What are the benefits of eating an apple a day?
            Coherence: 0.8666666666666667
            Relevance: 0.09183443807042975
            Redundancy: 0.0

