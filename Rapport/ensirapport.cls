%------------------------------------------------------------------------------%
%                                                                              %
%                                                                              %
%   EnsiRapport                                                                %
%                                                                              %
%   Version : 1.0                                                              %
%                                                                              %
%   Auteur : Arthur Sonzogni                                                   %
%                                                                              %
%------------------------------------------------------------------------------%


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ensirapport}[2014/01/01 classe pour des rapport à l'ensimag]

\typeout{}
\typeout{-----[ Ensirapport <debut> ]------}
\typeout{}

% classe de base

\LoadClassWithOptions{article}

% extensions

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{graphicx}
\RequirePackage[french]{babel}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{eurosym}
\RequirePackage{xcolor}

\DeclareOption{margeCorrection}{
	\AtEndOfClass{
		\RequirePackage[margin=1in]{geometry}
	}
}

% modification style des liens
\DeclareOption{liens}{
	\AtEndOfClass{
		\RequirePackage{hyperref}
        \hypersetup{ 
            colorlinks=true,
            breaklinks=true,
            urlcolor= black, 
            linkcolor= black, 
            citecolor= black
            }
	}
}
\ProcessOptions

% une commande pour ne pas que maketitle efface ces traces (@title,@date,@author)
% c'est une commande privée du package
\newcommand{\preserveMaketitle}{
	\let\oldmaketitle\maketitle
	\renewcommand{\maketitle}{
		\let\@preservedTitle\@title
		\let\@preservedAuthor\@author
		\let\@preservedDate\@date
		\oldmaketitle
		\let\@title\@preservedTitle
		\let\@author\@preservedAuthor
		\let\@date\@preservedDate
	}
}

% config fancyhdr
\DeclareOption{entete}{
	\AtEndOfClass{
		\RequirePackage{fancyhdr}
		\preserveMaketitle
		\RequirePackage{wallpaper}
		\RequirePackage{lastpage}

		\pagestyle{fancy}

		\fancyhf{}

		\renewcommand\headrule{
			\begin{minipage}{1\textwidth}
				\hrule width \hsize \kern 1mm \hrule width \hsize height 2pt 
			\end{minipage}
		}

		\renewcommand{\headrulewidth}{1pt}
		\renewcommand{\footrulewidth}{1pt}
		\lhead{ \textbf{\@author}}
		\chead{ \nouppercase\leftmark}
		\rhead{ \@date }
		\rfoot{ \textbf{page \thepage /\pageref*{LastPage}} }

		\fancypagestyle{plain}{ %
			\fancyhf{}
			\renewcommand{\headrulewidth}{0pt}
			\renewcommand{\footrulewidth}{1pt}
			\lhead{ \textbf{\@author}}
			\rhead{ \@date }
			\rfoot{ \textbf{page \thepage  /\pageref*{LastPage} } }
		}

	}
}
\ProcessOptions

% config entete-ensimag
\DeclareOption{entete-ensimag}{
	\AtEndOfClass{
		\RequirePackage{fancyhdr}
		\preserveMaketitle
		\RequirePackage{lastpage}

		\pagestyle{fancy}

		\fancyhf{}
		
		\def\headheightone{1pt}
		\def\headheighttwo{2pt}

		\renewcommand\headrule{
			\begin{minipage}{1\textwidth}
				\hrule width \hsize height \headheightone
				%\kern 1mm
				%\hrule width \hsize height \headheighttwo
			\end{minipage}
		}

		\addtolength{\headheight}{111pt}
		\addtolength{\textheight}{-111pt}
		\renewcommand{\headrulewidth}{1pt}
		\renewcommand{\footrulewidth}{1pt}
		\lhead{ 
			\includegraphics[height=70pt]{ensilogo.png}
		}
		\chead{
            \textbf{\@author} ~,~ \@date
		}
		\rhead{
            %\includegraphics[height=70pt]{ljk.png}
        }
		\rfoot{ \textbf{page \thepage  /\pageref*{LastPage}} }


		\fancypagestyle{plain}{ %
			\fancyhf{}
			\renewcommand{\headheightone}{0pt}
			\renewcommand{\headheighttwo}{0pt}
			\renewcommand{\footrulewidth}{0pt}
			\lhead{
				\includegraphics[height=90pt]{ensilogo.png}
			}
			\chead{}
			\rhead{
                %\includegraphics[height=90pt]{ljk.png}
            }
			%\rfoot{ \textbf{page \thepage /\pageref*{LastPage}} }
		}

	}
}
\ProcessOptions

% minted
\DeclareOption{minted}{
	\AtEndOfClass{
		\RequirePackage{minted}
	}
}
\ProcessOptions

% permet d'include directement des fichiers eps.
\RequirePackage{epstopdf}


\typeout{}
\typeout{-----[ Ensirapport <fin> ]------}
\typeout{}
