doc: $(shell find docs) $(shell find pyfluo)
	cd docs
	git add .
	git commit -m "updated docs"
	make html
	cd ../gh-pages/html
	git add .
	git commit -m "rebuilt docs"
	git push
	cd ../..
	git add gh-pages/html
	git commit -m "updated gh-pages"
	git push
