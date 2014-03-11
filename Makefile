doc: docstrings pages docedits ghp

docstrings: $(shell find pyfluo) $(shell find docs)
	cd pyfluo  &&\
	git add -A .

pages: $(shell find docs)
	cd docs &&\
	git add -A . &&\
	git commit -m "updated docs"

docedits: docstrings pages
	cd docs &&\
	make html

ghp: docedits
	cd gh-pages/html &&\
	git add -A . &&\
	git commit -m "rebuilt docs" &&\
	git push &&\
	cd ../.. &&\
	git add gh-pages/html &&\
	git commit -m "updated gh-pages" &&\
	git push

