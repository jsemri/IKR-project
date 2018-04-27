NAME=ikr

$(NAME).pdf: $(NAME).ps
	ps2pdf -dPDFSETTINGS=/prepress $(NAME).ps

$(NAME).ps: $(NAME).dvi
	dvips  $(NAME).dvi

$(NAME).dvi: $(NAME).tex
	latex $(NAME).tex
	latex $(NAME).tex

pack:
	zip xsemri00-fit.zip $(NAME).tex makefile *.eps

clean:
	-rm $(NAME).dvi $(NAME).aux $(NAME).toc $(NAME).log $(NAME).out

