# Person B targets
.PHONY: b_rag b_eval b_sent b_sent_eval b_kpi b_anom b_ui

b_rag:
	@python scripts/rag_answer.py

b_eval:
	@python scripts/eval_qa.py

b_sent:
	@python scripts/sentiment_pipeline.py

b_sent_eval:
	@python scripts/sentiment_eval.py

b_kpi:
	@python scripts/kpi_extract.py

b_anom:
	@python scripts/anomalies_rules.py

b_ui:
	streamlit run app/ui_app.py
