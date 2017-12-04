results/%bench.json:
	python $(patsubst %.json,mlbench/%.py, $(notdir $@)) > $@

clean:
	rm results/*.json
