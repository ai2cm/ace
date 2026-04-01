ENVIRONMENT_NAME = fme
LOCAL_DIR = /tmp/aimip-ace/

postprocess:
	conda run -n $(ENVIRONMENT_NAME) python postprocess.py

test:
	conda run -n $(ENVIRONMENT_NAME) python -m pytest test_postprocess.py -v --noconftest

clean:
	rm -rf $(LOCAL_DIR)

.PHONY: postprocess test clean
