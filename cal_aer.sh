#!/bin/bash




export WORKLOC=/home/acc_align
datadir=$WORKLOC/data/deen

ref_align=$datadir/deen.talp
reftype='--oneRef'



for LayerNum in `seq 1 12`; do
    echo "=====AER shifted for de-en layer=${LayerNum}..."
    python $WORKLOC/aer.py ${ref_align} $WORKLOC/xxx/de2en.align.$LayerNum --fAlpha 0.5 $reftype
done








