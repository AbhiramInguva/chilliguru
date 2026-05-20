# ChilliGuru shortcuts
# Usage: make test | make test-live | make deploy | make upload

.PHONY: test test-live deploy upload clean status

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
