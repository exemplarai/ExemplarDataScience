import pandas as pd



output_spotify_play =       pd.read_csv("/home/stas/hdd/coding/python/data-export.exemplar.ai/output_spotify_play.csv")
output_square_payments =    pd.read_csv("/home/stas/hdd/coding/python/data-export.exemplar.ai/output_square_payments.csv")
output_square_locations =   pd.read_csv("/home/stas/hdd/coding/python/data-export.exemplar.ai/output_square_locations.csv")


output_spotify_play['key'] = output_spotify_play['key'].str.replace("Jesus", "8BSTTGBX5Z7VM")


output_spotify_play[output_spotify_play["key"] == "8BSTTGBX5Z7VM"].to_csv("output_spotify_play_8BSTTGBX5Z7VM.csv", index=False)
output_spotify_play[output_spotify_play["key"] == "FM9W3RHMJFQ5M"].to_csv("output_spotify_play_FM9W3RHMJFQ5M.csv", index=False)


output_square_payments[output_square_payments["location"] == "8BSTTGBX5Z7VM"].to_csv("output_square_payments_8BSTTGBX5Z7VM.csv", index=False)
output_square_payments[output_square_payments["location"] == "FM9W3RHMJFQ5M"].to_csv("output_square_payments_FM9W3RHMJFQ5M.csv", index=False)


# output_spotify_play['key']
# FM9W3RHMJFQ5M
# Jesus

# output_square_payments['location']
# FM9W3RHMJFQ5M
# 8BSTTGBX5Z7VM

