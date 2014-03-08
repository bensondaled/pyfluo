doc: pages docstrings ghp

docstrings: $(shell find pyfluo) $(shell find docs)
	cd docs &&\
	make html

pages: $(shell find docs)
	cd docs &&\
	git add -A . &&\
	git commit -m "updated docs"

ghp: $(shell find docs)
	cd gh-pages/html &&\
	git add -A . &&\
	git commit -m "rebuilt docs" &&\
	git push &&\
	cd ../.. &&\
	git add gh-pages/html &&\
	git commit -m "updated gh-pages" &&\
	git push

