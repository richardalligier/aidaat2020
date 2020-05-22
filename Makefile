
include config

PYTHON=python3

HYPERPARAMETERS_TO_TEST=100

export PATH := $(shell pwd)/OCaml/bin:$(PATH)

.PHONY: all tables figures
.SECONDARY:

SETS := train valid test
$(DATA_PATH)/%:	MODEL = $(wordlist 1,1, $(subst /, ,$*))
$(DATA_PATH)/%:	YVAR = $(wordlist 2,2, $(subst /, ,$*))
$(DATA_PATH)/%:	METH = $(wordlist 3,3, $(subst /, ,$*))

OPENSKYURL=https://opensky-network.org/datasets/publication-data/climbing-aircraft-dataset

all: PREDICTED


PREDICTED: $(foreach model, $(MODELS),$(foreach yvar, $(YVARS), $(foreach meth, $(METHODS), $(DATA_PATH)/$(model)/aidaat2020$(yvar)/$(meth)/predicted.csv)))


$(DATA_PATH)/trajs/%:
	wget $(OPENSKYURL)/trajs/$* -P $(@D)

$(DATA_PATH)/foldedtrajs/%_test.csv.xz: $(DATA_PATH)/trajs/%_test.csv.xz
	@echo "============= Building test example set $@ ============="
	@make -C ./OCaml all
	@mkdir -p $(DATA_PATH)/foldedtrajs
	xzcat $(DATA_PATH)/trajs/$*_test.csv.xz | csvaddenergyrate -alt baroaltitudekalman -tas taskalman -temp tempkalman | csvfold -c energyrate -npast 1:2:3:4:5:6:7:8:9 | csvremove -c energyrate -all | csvfold -c baroaltitudekalman -npast 1:2:3:4:5:6:7:8:9 | csvfold -c taskalman -npast 1:2:3:4:5:6:7:8:9 | csvfold -c vertratecorr -npast 1:2:3:4:5:6:7:8:9 | csvaddfeatures | csvfold -c baroaltitude -nfutur 1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25:26:27:28:29:30:31:32:33:34:35:36:37:38:39:40 | xz -1 > $@

$(DATA_PATH)/foldedtrajs/%.csv.xz: $(DATA_PATH)/trajs/%.csv.xz
	@echo "============= Building train/valid example set $@ ============="
	@make -C ./OCaml all
	@mkdir -p $(DATA_PATH)/foldedtrajs
	xzcat $(DATA_PATH)/trajs/$*.csv.xz  | csvaddenergyrate -alt baroaltitudekalman -tas taskalman -temp tempkalman | csvfold -c energyrate -npast 1:2:3:4:5:6:7:8:9 | csvremove -c energyrate -all | csvfold -c baroaltitudekalman -npast 1:2:3:4:5:6:7:8:9 | csvfold -c taskalman -npast 1:2:3:4:5:6:7:8:9 | csvfold -c vertratecorr -npast 1:2:3:4:5:6:7:8:9 | csvfold -c baroaltitude -nfutur 40 | csvremove -c baroaltitudep40 -empty | csvremove -c baroaltitudekalmanm9 -empty | csvaddfeatures | xz -1 > $@

.SECONDEXPANSION:
# computing the performance the obtained models for various hyperparameters

$(DATA_PATH)/paramlist:
	$(PYTHON) ./Python/generateparamlist.py -pca -whiten > $@

$(DATA_PATH)/%/hyperparameters: $(DATA_PATH)/foldedtrajs/$$(MODEL)_valid.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_train.csv.xz $(DATA_PATH)/paramlist
	@mkdir -p $(DATA_PATH)/$*/loghyperparameters
	@echo "===Searching for hyperparameter: $(MODEL) $(YVAR) $(XVAR)  $(METH) ==="
	$(PYTHON) ./Python/batchtrain.py -model $(MODEL) -method $(METH) -xvars $(XVAR) -nworker $(NWORKER_BY_GPU) -ngpu $(NGPU) -folderlogout $(DATA_PATH)/$*/loghyperparameters -iend $(HYPERPARAMETERS_TO_TEST) -resume -batch $(DATA_PATH)/paramlist
	$(PYTHON) ./Python/selecthyper.py -loghyperparameters $(DATA_PATH)/$*/loghyperparameters  > $(DATA_PATH)/$*/hyperparameters



$(DATA_PATH)/%/samediag/predicted.csv:  $(DATA_PATH)/foldedtrajs/$$(MODEL)_valid.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_train.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_test.csv.xz
	@mkdir -p $(@D)
	$(PYTHON) ./Python/samediagfull.py -method diag -fileout $@ -model $(MODEL) > $(@D)/logfinal

$(DATA_PATH)/%/samefull/predicted.csv:  $(DATA_PATH)/foldedtrajs/$$(MODEL)_valid.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_train.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_test.csv.xz
	@mkdir -p $(@D)
	$(PYTHON) ./Python/samediagfull.py -method full -fileout $@ -model $(MODEL) > $(@D)/logfinal


$(DATA_PATH)/%/gbdiag/predicted.csv:  $(DATA_PATH)/foldedtrajs/$$(MODEL)_valid.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_train.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_test.csv.xz
	@mkdir -p $(@D)
	$(PYTHON) ./Python/samediagfull.py -method gbdiag -fileout $@ -model $(MODEL) > $(@D)/logfinal

$(DATA_PATH)/%/gbfull/predicted.csv:  $(DATA_PATH)/foldedtrajs/$$(MODEL)_valid.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_train.csv.xz $(DATA_PATH)/foldedtrajs/$$(MODEL)_test.csv.xz
	@mkdir -p $(@D)
	$(PYTHON) ./Python/samediagfull.py -method gbfull -fileout $@ -model $(MODEL) > $(@D)/logfinal

# training the model on the first ten month
$(DATA_PATH)/%/predicted.csv: $(DATA_PATH)/%/hyperparameters $(DATA_PATH)/foldedtrajs/$$(MODEL)_test.csv.xz
	@echo "inside hyperparameter=>predictions"
	@echo $(MODEL) $(YVAR) $(XVAR)
	@mkdir -p $(DATA_PATH)/$*/models
	$(PYTHON) ./Python/batchtrain.py -model $(MODEL) -method $(METH) -nworker $(NWORKER_BY_GPU) -ngpu $(NGPU) -foldermodelout $(DATA_PATH)/$*/models -batch $(DATA_PATH)/$*/hyperparameters -finalmodel
	@mkdir -p $(@D)
	$(PYTHON) ./Python/test.py -foldermodel $(DATA_PATH)/$*/models -fileout $@ > $(@D)/logfinal

tables:
	@mkdir -p $(TABLE_FOLDER)
	$(PYTHON) ./Python/computestats.py -which tables

