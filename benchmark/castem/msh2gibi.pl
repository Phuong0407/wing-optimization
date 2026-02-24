#!/usr/bin/perl
# msh2gibi.pl         SOURCE    LC    01/10/24
#
# [][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][]
# CHAMPANEY Laurent  Universite de Versailles St Quentin le 24 / 10 / 01
#
# Transfert de maillages GMSH vers GIBIANE
#
#  Usage :
#    msh2gibi.pl [-d densite] [-n dimension] [-v] mail
#  
#  lit mail.msh et le converti en mail.sauv
#
#  Utilisation dans CAST3M (Gibiane)
#    OPTI REST FORMAT 'mail.sauv'; REST FORMAT;
#
# ======================================================================
#
# ======================================================================
# Analyse des arguments
# ======================================================================
use Getopt::Long;
use Tk ;
use Tk::FileSelect ;
use Tk::ROText;
#
# Lecture des options {{{
@knownoptions = (
                 "mode|m=s",
                 "help|h",
                 "dimension|n=i",
                 "densite|d=f",
                 "physical|p",
                 "verbose|v");
GetOptions (@knownoptions) || exit ;
$ndime1 = 2;
if ($opt_dimension){$ndime1=$opt_dimension};
$densi1 = 1.;
if ($opt_densite){$densi1=$opt_densite};
$verb1 = 0;
if ($opt_verbose){$verb1=1};
$phys1 = 0;
if ($opt_physical){$phys1=1};
#
if ($opt_help){sortie_usage(); exit();};
# }}}
#
if($#ARGV==-1) { 
        xmsh2gibi();
} else {
        $file1 = $ARGV[0];
        msh2gibi();
}
#
exit;

#
sub xmsh2gibi {{{
$main_win = MainWindow->new ;

load_usage();

# Fenetre du menu
# ~~~~~~~~~~~~~~~	

# Title for the main window
$main_win->title( "Conversion GMSH vers CAST3M" ) ;
 
# Fichier 
$main_win->Label( -text => "Fichier .msh :")
  -> grid( -row => 1, -column => 0 , 
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;

$main_win->Entry( -textvariable => \$file1,-width => 35)
  -> grid( -row => 1, -column => 1 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;
$op_select = $main_win->Button( -text => "...",
				  -command => \&open_file )
  -> grid( -row => 1, -column => 2 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;

#
# Mode 
$main_win->Label( -text => "Mode")
  -> grid( -row => 2, -column => 0 , 
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'e' ) ;
$mode_displayed = "Automatique";
$main_win->Optionmenu(-options=>[["Automatique",""],
				 ["Déformations Planes","DP" ],
				 ["Contraintes Planes", "CP"],
				 ["Axisymétrique", "AX"],
				 ["Tridimensionnel", "3D"]],
			-textvariable => $mode_displayed,
		       -variable=>\$opt_mode)
  -> grid( -row => 2, -column => 1 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;
#
#
# Mode 
$main_win->Label( -text => "Densité")
  -> grid( -row => 3, -column => 0 , 
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'e' ) ;
$main_win->Entry( -textvariable => \$densi1,-width => 5)
  -> grid( -row => 3, -column => 1 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;
# Conversion
$op_sort_bt = $main_win->Button( -text => "Conversion",
				  -command => \&convert )
  -> grid( -row => 4, -column => 1 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'e' ) ;

$main_win->Checkbutton( -text => "Physical",
			variable => \$phys1,
			-command => 
			sub { } )
  -> grid( -row => 4, -column => 0 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'e' ) ;

# Help 
$main_win->Button( -text => "Help",
				  -command => \&appel_help )
  -> grid( -row => 3, -column => 2 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'nsew' ) ;

# Fin 
$main_win->Button( -text => "Sortie",
				  -command => sub{exit} )
  -> grid( -row => 4, -column => 2 ,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'w' ) ;
# Copyright
$main_win->Label( -text => "[msh2gibi.pl - L.CHAMPANEY - LEMA/UVSQ - 2003]")
  -> grid( -row => 5, -column => 0 , -columnspan => 3,
	   -pady => 4,
	   -padx => 2,
	   -sticky => 'nsew' ) ;

# Help
$help_win = MainWindow->new ;
$help_win->resizable(0,0) ;
$help_win->geometry("+0-20") ;
$help_win->withdraw() ;
$help_txt = $help_win->Scrolled("ROText",
			-width=>56,
			-height=>25) -> pack(-expand=>1);
$help_win->Button( -text => "Sortie",
                                  -command => \&help_fin )
  ->  pack(-expand=>1);

$help_txt->insert('end',$help_text);

#
sub appel_help {
      $help_win->deiconify() ;
      $help_win->raise() ;
      return;
}; 
sub help_fin {
      $help_win->withdraw() ;
      return;
} ;
#
# Pour l'ouverture
$open_FS = $main_win -> FileSelect(-filter => "*.msh");
sub open_file { $file1 = $open_FS -> Show; }
# Appel à la conversion
sub convert { if ($file1){ msh2gibi(); $file1 = ""; } }
#
# Activate the Main window
$main_win->deiconify() ;
$main_win->raise() ;
# -----------------------------------------------------> Main loop <---
MainLoop ;
#
return;
#
}}};
#
sub msh2gibi {{{
# 
# Les options
#
$ifour1 = -1;
$ifomod1 = -1;
if ($opt_mode){
	$ifour1 = -1; $ndime1 = 2;
	if ($opt_mode eq 'DP') {$ifour1 = -1; $ifomod1 = -1; $ndime1 = 2};
	if ($opt_mode eq 'CP') {$ifour1 = -2; $ifomod1 = -1; $ndime1 = 2};
	if ($opt_mode eq 'AX') {$ifour1 =  0; $ifomod1 =  0; $ndime1 = 2};
	if ($opt_mode eq '3D') {$ifour1 =  2; $ifomod1 = 2; $ndime1 = 3};
}else {
	$ifour1 = -1; $ndime1 = 2; $ifomod1 = -1;
};
#
# ======================================================================
# Fichier d'ecriture vers Castem
# ======================================================================
($generic , $generic_ext) = ( $file1 =~ /([-_:\/\w]*)\.?(\w*)/ ) ;
$msh_file = "$generic.msh";
$gibi_file ="$generic.sauv";
$phys_file = "$generic.phy";
if ($phys1 && (-e $phys_file)) {
	print "Entitees Physiques :\n";
	open (PHYSFIC, $phys_file) or die "Le fichier $phys_file n'extiste pas";
	while ($line = <PHYSFIC>) {
		$line =~ /^\s*(\S+)\s*=\s*([0-9]*)\s*;/ ;
		$nom=$1;
		$nom=~ tr/a-z/A-Z/;
		$Phys[$2] = $nom;
		print "$1=$nom\n";
	}
}
#
# ======================================================================
# En premier on teste la dimension du probleme
# ======================================================================
open (INFIC, $msh_file) || die "Le fichier $msh_file n existe pas\n";
$line = <INFIC>;
chomp($line);
if ($line eq '$NOD') {
	if ($verb){print "Recherche de la dimension\n";}
}
else {
	print "Il ne s'agit pas d'un fichier gmsh\n";
	exit;
}
#
$line = <INFIC>;
chomp($line);
$nb_node1=int($line);
if ($verb1) {print "$nb_node1 noeuds dans le modele\n";}
#
  for ($i = 0 ; $i <= $nb_node1 ; $i++) {
		$line = <INFIC>;
		@tab = split ( /\s+/ , $line ) ;
		$corres[@tab[0]]=$i+1;
                $coor_z = @tab[3];
		if ($coor_z != 0.) { $ndime1 = 3; $ifour1 = 2, $ifomod1 = 2 }
	}
close (INFIC);
#
if ($verb1) {print "Dimension : $ndime1\n";}
#
# ======================================================================
# On s'occupe du fichier des noeuds
# ======================================================================
open (INFIC, $msh_file);
$line = <INFIC>;
$line = <INFIC>;
#
open (NODEFIC, ">gms2gibi.temp.node");
#
print NODEFIC " ENREGISTREMENT DE TYPE   2\n";
printf NODEFIC (" PILE NUMERO  32NBRE OBJETS NOMMES       0NBRE OBJETS%8d\n", $nb_node1);
printf NODEFIC ("%8d\n", $nb_node1);
$nli1 = int($nb_node1 / 10);
$nre1 = $nb_node1 - (10 * $nli1);
#
if ($nli1 > 0)
	{
	for ($i = 0; $i < $nli1; $i++)
		{
		for ($k = 1; $k <= 10; $k++)
			{
			printf NODEFIC ("%8d",(($i * 10) + $k));
			}
		print NODEFIC "\n";
		}
	}
if ($nre1 > 0)
        {
        for ($k = 1; $k <= $nre1; $k++)
	        {
	        printf NODEFIC ("%8d",(($nli1 * 10) + $k));
	        }
        print NODEFIC "\n";
        }
#
print  NODEFIC " ENREGISTREMENT DE TYPE   2\n";
print  NODEFIC " PILE NUMERO  33NBRE OBJETS NOMMES       0NBRE OBJETS       1\n";
printf NODEFIC ("  %6d\n", (($ndime1 + 1) * $nb_node1));
$ind1 = 0;
for ($i = 0 ; $i < $nb_node1 ; $i++)
	{
	$line = <INFIC>;
	@tab = split ( /\s+/ , $line ) ;
#       Les coordonnees
        for ($k = 1; $k <= $ndime1; $k++)
	 	{
#		if (@tab[$k] >= 0.)
#		{
         	printf NODEFIC ("%+22.14E", @tab[$k]);
#         	printf NODEFIC ("%22.14E", @tab[$k]);
#         	printf NODEFIC ("%3E22.14", @tab[$k]);
#		}
#		else
#		{
#         	printf NODEFIC (" %+20.14E", @tab[$k]);
#		}
	        $ind1++;
		if ($ind1 >= 3)
			{
			$ind1 = 0;
         	        print NODEFIC "\n";
			}
		}
#       La densite
       	printf NODEFIC ("%+22.14E", $densi1);
        $ind1++;
	if ($ind1 >= 3)
		{
		$ind1 = 0;
       	        print NODEFIC "\n";
		}

	}	
if ($ind1 != 0) {print NODEFIC "\n"};
#
print NODEFIC " ENREGISTREMENT DE TYPE   5\n";
printf NODEFIC ("LABEL AUTOMATIQUE :   %-50s","1");
#
close (NODEFIC);
#
# ======================================================================
# On s'occupe du fichier des maillages
# ======================================================================
$line = <INFIC>;
$line = <INFIC>;
$line = <INFIC>;
chomp($line);
$nb_elem1 = $line;
if ($verb1) {print "$nb_elem1 au total\n";}
#
open (ELEMFIC1, ">gms2gibi.temp.elem");
#
# Petits tableaux de travail
# corresp en type elemens
$type_elem[1] = 2;
$type_elem[2] = 4;
$type_elem[3] = 8;
$type_elem[4] = 23;
$type_elem[5] = 14;
$type_elem[6] = 16;
$type_elem[7] = 25;
$type_elem[15] = 1;
# corresp en nombre de noeuds
$nnod_elem[1] = 2;
$nnod_elem[2] = 3;
$nnod_elem[3] = 4;
$nnod_elem[4] = 4;
$nnod_elem[5] = 8;
$nnod_elem[6] = 6;
$nnod_elem[7] = 5;
$nnod_elem[15] = 1;
# corresp en type d'entite
$enti_elem[1] = 'L';
$enti_elem[2] = 'S';
$enti_elem[3] = 'S';
$enti_elem[4] = 'V';
$enti_elem[5] = 'V';
$enti_elem[6] = 'V';
$enti_elem[7] = 'V';
$enti_elem[15] = 'P';
#
$nb_maillage = 0;
$nb_maillag2 = 0;
#
$i_coul = 0;
#
$iobj1 = 0;
$index1[1] = 0;
$index1[2] = 0;
$index1[3] = 0;
$index1[4] = 0;
$nb_elem2[1] = 0;
$nb_elem2[2] = 0;
$nb_elem2[3] = 0;
$nb_elem2[4] = 0;
$nb_type1 = 0;
#
for ($i = 0 ; $i < $nb_elem1 ; $i++)
	{
	$line = <INFIC>;
	@tab = split ( /\s+/ , $line ) ;
#       L'objet
        $jobj1 = @tab[2];
	if ($iobj1 == 0) {$iobj1 = $jobj1};
# On teste si on change d'objet
	if ($jobj1 != $iobj1)
		{
#               Nouveau maillage : on ecrit 
		ecrit_maillage();
#
        	$iobj1 = $jobj1;
                $index1[1] = 0;
                $index1[2] = 0;
                $index1[3] = 0;
                $index1[4] = 0;
                $nb_elem2[1] = 0;
                $nb_elem2[2] = 0;
                $nb_elem2[3] = 0;
                $nb_elem2[4] = 0;
                $nb_type1 = 0;
		}
#
        if ($nb_type1 == 0)
                {
                $nb_type1++;
	        $type_elem1[1]=@tab[1];
                $nb_elem2[1]++;
                $index1[1]++;
                $list_ligne[1][$index1[1]] = $line;
                }
         else
                {
                $test1 = 0;
                for ($j = 1 ; $j <= $nb_type1 ; $j++)
                        {
                        if (@tab[1] == $type_elem1[$j]) 
                                {
                                $nb_elem2[$j]++;
                                $index1[$j]++;
                                $list_ligne[$j][$index1[$j]] = $line;
                                $test1 = 1; 
                                }
                        }
                if ($test1 == 0)
                        {
                        $nb_type1++;
                        $type_elem1[$nb_type1]=@tab[1];
                        $nb_elem2[$nb_type1]++;
                        $index1[$nb_type1]++;
                        $list_ligne[$nb_type1][$index1[$nb_type1]] = $line;
                        }
                }
#
#       Les numeros de noeud

	}	
# Dernier maillage : on ecrit 
ecrit_maillage();
close(ELEMFIC1);
#
if ($verb1) {print "$nb_maillag2 maillages dont $nb_maillage nommes\n";}
#
#  On ecrit tout
open (OUFIC, ">$gibi_file");
# Options renerales
print OUFIC " ENREGISTREMENT DE TYPE   4\n";
printf OUFIC (" NIVEAU  12 NIVEAU ERREUR   0 DIMENSION%4d\n",$ndime1);
print OUFIC " DENSITE 0.10000E+01\n";
print OUFIC " ENREGISTREMENT DE TYPE   7\n";
print OUFIC " NOMBRE INFO CASTEM2000   8\n";
printf OUFIC (" IFOUR%4d NIFOUR   0 IFOMOD%4d IECHO   1 IIMPI   0 IOSPI   0 ISOTYP   1\n", $ifour1, $ifomod1);
print OUFIC " NSDPGE     0\n";
print OUFIC " ENREGISTREMENT DE TYPE   2\n";
# Elements
printf OUFIC (" PILE NUMERO   1NBRE OBJETS NOMMES%8dNBRE OBJETS%8d\n",$nb_maillage,$nb_maillag2);
$npoin1 = 0;
$nline1 = 0;
$nsurf1 = 0;
$nvolu1 = 0;
$ind1 = 0;
for ($i = 1 ; $i <= $nb_maillage ; $i++)
	{
	if ($type_mail[$i] eq 'P')
		{
                $npoin1++;
		$nom = "$type_mail[$i]$npoin1";
		}
	if ($type_mail[$i] eq 'L')
		{
                $nline1++;
		$nom = "$type_mail[$i]$nline1";
		}
	if ($type_mail[$i] eq 'S')
		{
                $nsurf1++;
		$nom = "$type_mail[$i]$nsurf1";
		}
	if ($type_mail[$i] eq 'V')
		{
                $nvolu1++;
		$nom = "$type_mail[$i]$nvolu1";
		}
	if ($Phys[$num_entit[$i]]) {$nom = $Phys[$num_entit[$i]]};
	printf OUFIC (" %-8s",$nom);
        $ind1++;
	if ($ind1 >= 8)
		{
		$ind1 = 0;
       	        print OUFIC "\n";
		}
	}
if ($ind1 != 0) {print OUFIC "\n"};
$ind1 = 0;
for ($i = 1 ; $i <= $nb_maillage ; $i++)
	{
	printf OUFIC ("%8d",$num_mail[$i]);
        $ind1++;
	if ($ind1 >= 10)
		{
		$ind1 = 0;
       	        print OUFIC "\n";
		}
	}
if ($ind1 != 0) {print OUFIC "\n"};
#
# On concatene
open (INFIC, "gms2gibi.temp.elem");
while ($line=<INFIC>) {
	print OUFIC "$line";
};
close (INFIC);
open (INFIC, "gms2gibi.temp.node");
while ($line=<INFIC>) {
	print OUFIC "$line";
};
close (INFIC);
#
close (OUFIC);
#
#
# On fait le menage
unlink <gms2gibi.temp.elem>;
unlink <gms2gibi.temp.node>;
#
# ======================================================================
# On sort
# ======================================================================
print "\n";
print "Conversion $msh_file vers $gibi_file effectuee\n";
print "   (Densité $densi1 - Dimension : $ndime1)\n";
print "Restitution :    OPTI REST FORMAT '$gibi_file'; REST FORMAT; \n";
print "\n";
#
return;
#
}}} ;
#
sub ecrit_maillage {{{
        if ($nb_type1 > 1)
                {
#               Nouvel objet : on enregistre
                if ($verb1) {print "Objet complexe : $nb_type1 types d elements\n";}
#               Entete
                printf ELEMFIC1 ("%8d",0);
                printf ELEMFIC1 ("%8d",$nb_type1);
                printf ELEMFIC1 ("%8d",0);
                printf ELEMFIC1 ("%8d",0);
                printf ELEMFIC1 ("%8d\n",0);
#
                $nb_maillage++;
		$i_coul++;
		if ($icoul == 7) {$icoul = 0};
                $type_mail[$nb_maillage] = $enti_elem[$type_elem1[1]];
		$num_entit[$nb_maillage] = $iobj1;
                $num_mail[$nb_maillage] = $nb_maillag2 + 1;
#
                for ($k = 1; $k <= $nb_type1; $k++)
                        {
                        printf ELEMFIC1 ("%8d",($nb_maillag2 + $k + 1));
                        }
                print ELEMFIC1 "\n";
                $nb_maillag2 = $nb_maillag2 + $nb_type1 + 1; 
                }
        else                
                {
                $nb_maillage++;
		$i_coul++;
		if ($icoul == 7) {$icoul = 0};
                $nb_maillag2++;
                $type_mail[$nb_maillage] = $enti_elem[$type_elem1[1]];
		$num_entit[$nb_maillage] = $iobj1;
                $num_mail[$nb_maillage] = $nb_maillag2;
                }
#
        for ($h = 1; $h <= $nb_type1; $h++)
                {
#               Nouvel objet : on enregistre
                $inum1 = $nb_maillag2 - $nb_type1 + $h;
                if ($verb1) {print "Maillage $inum1 : $nb_elem2[$h] elements\n";}
#               Entete
                printf ELEMFIC1 ("%8d",$type_elem[$type_elem1[$h]]);
                printf ELEMFIC1 ("%8d",0);
                printf ELEMFIC1 ("%8d",0);
                printf ELEMFIC1 ("%8d",$nnod_elem[$type_elem1[$h]]);
                printf ELEMFIC1 ("%8d\n",$nb_elem2[$h]);
#               Couleurs
                $ind2 = 0;
                for ($k = 1; $k <= $nb_elem2[$h]; $k++)
                        {
                        printf ELEMFIC1 ("%8d",$i_coul);
                        $ind2++;
                        if ($ind2 >= 10)
                                {
                                $ind2 = 0;
                                print ELEMFIC1 "\n";
                                }
                        }

                if ($ind2 != 0) {print ELEMFIC1 "\n"};
#               Noeuds                    
                $ind1 = 0;
                for ($j = 1; $j <= $nb_elem2[$h]; $j++)
                        {
                        $line2 = $list_ligne[$h][$j];
                        @tab2 = split ( /\s+/ , $line2 ) ;
                        $nno1 = @tab2[4];
                        for ($k = 5; $k <= (4 + $nno1); $k++)
                                {
                                printf ELEMFIC1 ("%8d", $corres[@tab2[$k]]);
                                $ind1++;
                                if ($ind1 >= 10)
                                        {
                                        $ind1 = 0;
                                        print ELEMFIC1 "\n";
                                        }
                                }
                        }
                if ($ind1 != 0) {print ELEMFIC1 "\n"};
                }
#
	return;
}}};
#
sub sortie_usage {{{
        load_usage();
        print "$help_text";
	return;
        }}};
#
sub load_usage {{{
        $help_text =
         "  \n".
         "========================================================  \n".
         "msh2gibi.pl : convertion GMSH vers CAST3M (Gibiane)\n".
         "Laurent CHAMPANEY                      Octobre 2001\n".
         "========================================================  \n".
         "  \n".
         "Usage :\n".
         "  msh2gibi.pl [-d densite] [-m mode] [-v] [-h] [-p] mail\n".
         "  \n".
         "  lit mail.msh et le converti en mail.sauv\n".
         "  \n".
         "  Restitution dans CAST3M\n".
         "    OPTI REST FORMAT 'mail.sauv'; REST FORMAT;\n".
         "  \n".
         "  L'option -h donne ces infos\n".
         "  L'option -v affiche des infos pendant l'execution\n".
         "  \n".
         "  Le mode de calcul dans cast3m est :\n".
         "    Deformations planes avec l'option -m DP\n".
         "    Contraintes  planes avec l'option -m CP\n".
         "    Axi-symetrique      avec l'option -m AX\n".
         "    Tridimensionnel     avec l'option -m 3D\n".
         "    Tridimensionnel     par defaut si il existe des\n".
         "                        points du maillage avec une\n".
         "                        coordonnée en z non nulle.\n".
         "    Deformations planes par defaut si il n'y a pas de\n".
         "                        point du maillage avec une\n".
         "                        coordonnée en z non nulle.\n".
         "  \n".
         "  Dans Cast3m les noms des maillages sont :\n".
         "    P1, P2, ... Pn : pour les points\n".
         "    L1, L2, ... Ln : pour les lignes\n".
         "    S1, S2, ... Sn : pour les surfaces\n".
         "    V1, V2, ... Vn : pour les volumes\n".
         "   avec l'option -p on lit un fichier mail.phy qui\n".
         "     contient les noms des maillages à récupérer dans \n".
         "     cast3M sous la forme :\n".
         "       lign1 = 10000;\n".
         "       surf1 = 10001;\n".
         "       volu1 = 10002\n".
         "     les noms sont les noms obtenus dans cast3m\n".
         "     et les numeros ceux des entites physiques\n".
         "     dans gmsh. On peut utiliser directement  \n".
         "     les noms des entites dans gmsh en ajoutant\n".
         "       Include \"mail.phy\";\n".
         "     en tete du fichier de geometrie (.geo) de gmsh.\n".
         "  \n".
         "========================================================  \n".
         "msh2gibi.pl : convertion GMSH vers CAST3M (Gibiane)\n".
         "Laurent CHAMPANEY                      Octobre 2001\n".
         "========================================================  \n".
         "  \n";
	return;
        }}};

