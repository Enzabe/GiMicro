BEGIN { FS=OFS=" " }
{ printf "%s%s", (FNR>1 ? OFS : "\n"), $ARGIND }
ENDFILE {
    print ""
    if (ARGIND < NF) {
        ARGV[ARGC] = FILENAME
        ARGC++
    }
}
