# ChilliGuru shortcuts
# Usage: make test | make test-live | make deploy | make upload | make validate-gate | make rollback-model | make purge-telemetry

.PHONY: test test-live deploy upload clean status validate-gate rollback-model purge-telemetry

test:
	python3 test_local.py

test-live:
	python3 test_live.py

deploy:
	./deploy.sh

upload:
	python3 upload_to_hf.py \
		--model chilli_pest_v2.pt \
		--info model_info.json \
		--app hf_space_app.py \
		--requirements hf_space_requirements.txt

clean:
	rm -rf __pycache__ kaggle_output/ runs/

status:
	git status
	@echo ""
	@echo "Local .pt files (should not be committed if gitignored):"
	@ls -la *.pt 2>/dev/null || echo "  (none in cwd)"

# Self-heal model-promotion gate -- see scripts/eval/run_validation_gate.py
validate-gate:
	python3 scripts/eval/run_validation_gate.py

# Roll back weights/chilli_pest_model.onnx to the previous registered version.
# Local-only: does not commit, push, or redeploy -- do that yourself after.
rollback-model:
	python3 scripts/rollback_model.py

# Delete shadow/telemetry farmer photos older than RETENTION_DAYS (dry-run by default).
purge-telemetry:
	python3 scripts/purge_old_telemetry.py
