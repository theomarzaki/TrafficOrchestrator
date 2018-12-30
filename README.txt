The source code, executable and JSON files are all required to run the Traffic Orchestration Simulation as specified in the Test Section of the Final Report Submission.

Please ensure that all dependencies mentioned in Section Test of the Final Report Submission is adhered to. The RapidJSON library is a header-only implementation. Please ensure that the ".../include/rapidjson" header folder is placed in the same directory as the source code. Once all dependencies are met:

1. Open a Terminal in the folder belonging to the source code. If the source code is moved to another file location, the Terminal must be opened there.

2. To compile and run the executable, the following commands are provided to the local directory terminal:

"g++ -std=c++11 -o exchange exchange.cpp"

followed by...

"./exchange"

The traffic orchestration will execute immediately requesting an input.

3. Refer back to the Testing / User Guide section for a walkthrough of the program.

4. Whilst following the User Guide, see "subscription_response.json" and "notify_small.json" in the Source Code provided.
