# idao-2020-final
Solution of team "Data O Plomo" to the final phase of the 2020 edition of the International Data Analysis Olympiad (IDAO)

## Track 2

```sh
rm -f track_2/*.csv track_2/*.dat
zip -jr submissions/track_2.zip track_2
```

### Performance profiling

```sh
cd track_2
pip install memory_profiler
mprof run python main.py
mprof plot --output ./results/track_2_memory_usage.png
```
