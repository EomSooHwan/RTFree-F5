# SAP dataset (dysarthric speech)
python eval_orig_sim.py \
    -t sap_dev \
    -d /data2/esyoon_hdd/soohwan/interspeech26/SAP/SpeechAccessibility_Research_Release \
    -n "[0,1,2,3]" \
    --sim_ckpt "/data2/esyoon_hdd/soohwan/interspeech26/checkpoints/UniSpeech/wavlm_large_finetune.pth" \
    --max_pairs_per_speaker 10

# L2-ARCTIC dataset (accented speech)
python eval_orig_sim.py \
    -t l2arctic \
    -d /data2/esyoon_hdd/soohwan/interspeech26/L2_ARCTIC \
    -n "[0,1,2,3]" \
    --sim_ckpt "/data2/esyoon_hdd/soohwan/interspeech26/checkpoints/UniSpeech/wavlm_large_finetune.pth" \
    --max_pairs_per_speaker 10