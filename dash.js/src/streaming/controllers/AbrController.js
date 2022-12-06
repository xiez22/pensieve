/**
 * The copyright in this software is being made available under the BSD License,
 * included below. This software may be subject to other third party and contributor
 * rights, including patent rights, and no such rights are granted under this license.
 *
 * Copyright (c) 2013, Dash Industry Forum.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *  * Neither the name of Dash Industry Forum nor the names of its
 *  contributors may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

import SwitchRequest from '../rules/SwitchRequest';
import BitrateInfo from '../vo/BitrateInfo';
import DOMStorage from '../utils/DOMStorage';
import ABRRulesCollection from '../rules/abr/ABRRulesCollection';
import MediaPlayerModel from '../models/MediaPlayerModel';
import FragmentModel from '../models/FragmentModel';
import EventBus from '../../core/EventBus';
import Events from '../../core/events/Events';
import FactoryMaker from '../../core/FactoryMaker';
import ManifestModel from '../models/ManifestModel';
import DashManifestModel from '../../dash/models/DashManifestModel';
import VideoModel from '../models/VideoModel';
import DashMetrics from '../../dash/DashMetrics';
import MetricsModel from '../models/MetricsModel';
import { logger } from '../../../externals/cea608-parser';
import * as ort from 'onnxruntime-web';

const ABANDON_LOAD = 'abandonload';
const ALLOW_LOAD = 'allowload';
const DEFAULT_VIDEO_BITRATE = 1000;
const DEFAULT_AUDIO_BITRATE = 100;
const QUALITY_DEFAULT = 0;
const S_INFO = 6, S_LEN = 8;
const TOTAL_VIDEO_CHUNKS = 48;
const HOST_IP = 'xz2000.cn';
const PORT = 12397;
//const dashMetrics = this.context.dashMetrics;
//const metricsModel = this.context.metricsModel;

function AbrController() {
    let context = this.context;
    let eventBus = EventBus(context).getInstance();
    let abrAlgo = -1;
    let bitrateArray = [200,300,480,750,1200,1850,2850,4300,5300];
    let reservoir = 5;
    let cushion = 10;
    let p_rb = 1;
    let pastThroughput = [];
    let pastDownloadTime = [];
    let bandwidthEstLog = [];
    let horizon = 5; // number of chunks considered
    let lastRequested = 0;
    let lastQuality = 1;
    let alpha = 12;
    let qualityLog = [];
    let switchUpCount = 0;
    let switchUpThreshold = [0,1,2,3,4,5,6,7,8,9];
    let p = 0.85;
    let lastIndex = -1;
    let instance,
        abrRulesCollection,
        rulesController,
        streamController,
        autoSwitchBitrate,
        topQualities,
        qualityDict,
        confidenceDict,
        bitrateDict,
        ratioDict,
        averageThroughputDict,
        streamProcessorDict,
        abandonmentStateDict,
        abandonmentTimeout,
        limitBitrateByPortal,
        usePixelRatioInLimitBitrateByPortal,
        manifestModel,
        dashManifestModel,
        videoModel,
        dashMetrics,
        metricsModel,
        mediaPlayerModel,
        domStorage,
        pensieveSession;

    // Pensieve
    let pensieveState = [];

    function setup() {
        autoSwitchBitrate = {video: true, audio: true};
        topQualities = {};
        qualityDict = {};
        confidenceDict = {};
        bitrateDict = {};
        ratioDict = {};
        averageThroughputDict = {};
        abandonmentStateDict = {};
        streamProcessorDict = {};
        limitBitrateByPortal = false;
        usePixelRatioInLimitBitrateByPortal = false;
        domStorage = DOMStorage(context).getInstance();
        mediaPlayerModel = MediaPlayerModel(context).getInstance();
        manifestModel = ManifestModel(context).getInstance();
        dashManifestModel = DashManifestModel(context).getInstance();
        videoModel = VideoModel(context).getInstance();
        dashMetrics = DashMetrics(context).getInstance();
        metricsModel = MetricsModel(context).getInstance();
        pensieveState = [];
        for (let i = 0; i < S_INFO; i++) {
            pensieveState.push(Array(S_LEN).fill(0.0));
        }
    }

    function initialize(type, streamProcessor) {
        streamProcessorDict[type] = streamProcessor;
        abandonmentStateDict[type] = abandonmentStateDict[type] || {};
        abandonmentStateDict[type].state = ALLOW_LOAD;
        eventBus.on(Events.LOADING_PROGRESS, onFragmentLoadProgress, this);
    }

    // returns size of last chunk using HTTPRequest object (not hardcoded :))
    function last_chunk_size(lastreq) {
        var tot = 0;
        for ( var tt = 0; tt < lastreq.trace.length; tt++ ) {
            tot = tot + lastreq.trace[tt]['b'][0];
        }
        return tot;
    }

    function next_chunk_size(index) {
        // video chunk sizes
        let size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658];
        let size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412];
        let size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850];
        let size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450];
        let size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954];
        let size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270];


        // upper number is 96 if 2 second chunks for weird video
        // if 4 second chunks, make that number 48
        // 64 for old video (racecar)
        if (index < 0 || index > 48) {
            return 0;
        }
        // var sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
        let result = [size_video6[index], size_video5[index], size_video4[index], size_video3[index], size_video2[index], size_video1[index]];
        return result
    }

    function getStabilityScore(b, b_ref, b_cur) {
        var score = 0,
        n = 0;
        if (lastIndex >= 1) {
            for (var i = Math.max(0, lastIndex + 1 - horizon); i<= lastIndex - 1; i++) {
            if (qualityLog[i] != qualityLog[i+1]) {
                n = n + 1;
            }
            }
        }
        if (b != b_cur) {
            n = n + 1;
        }
        score = Math.pow(2,n);
        return score;
    }

    function getEfficiencyScore(b, b_ref, w, bitrateArray) {
        var score = 0;
        score = Math.abs( bitrateArray[b]/Math.min(w, bitrateArray[b_ref]) - 1 );
        return score;
        }

        function getCombinedScore(b, b_ref, b_cur, w, bitrateArray) {
        var stabilityScore = 0,
        efficiencyScore = 0,
        totalScore = 0;
        // compute
        stabilityScore = getStabilityScore(b, b_ref, b_cur);
        efficiencyScore = getEfficiencyScore(b, b_ref, w, bitrateArray);
        totalScore = stabilityScore + alpha*efficiencyScore;
        return totalScore;  
    }

    function getBitrateFestive(prevQuality, bufferLevel, bwPrediction, lastRequested, bitrateArray) {
        var self = this, 
        bitrate = 0,
        tmpBitrate = 0,
        b_target = 0,
        b_ref = 0,
        b_cur = prevQuality,
        score_cur = 0,
        score_ref = 0;
        // TODO: implement FESTIVE logic
        // 1. log previous quality
        qualityLog[lastRequested] = prevQuality;
        lastIndex = lastRequested;
        // 2. compute b_target
        tmpBitrate = p*bwPrediction;
        for (var i = 9; i>=0; i--) { // todo: use bitrateArray.length
            if (bitrateArray[i] <= tmpBitrate) {
                b_target = i;
                break;
            }
            b_target = i;
        }
        // 3. compute b_ref
        if (b_target > b_cur) {
            switchUpCount = switchUpCount + 1;
            if (switchUpCount > switchUpThreshold[b_cur]) {
            b_ref = b_cur + 1;
            } else {
            b_ref = b_cur;
            }
        } else if (b_target < b_cur) {
            b_ref = b_cur - 1;
            switchUpCount = 0;
        } else {
            b_ref = b_cur;
            switchUpCount = 0; // this means need k consecutive "up" to actually switch up
        }
        // 4. delayed update
        if (b_ref != b_cur) { // need to switch
            // compute scores
            score_cur = getCombinedScore(b_cur, b_ref, b_cur, bwPrediction, bitrateArray);
            score_ref = getCombinedScore(b_ref, b_ref, b_cur, bwPrediction, bitrateArray);
            if (score_cur <= score_ref) {
            bitrate = b_cur;
            } else {
            bitrate = b_ref;
            if (bitrate > b_cur) { // clear switchupcount
                switchUpCount = 0;
            }
            }
        } else {
            bitrate = b_cur;
        }
        // 5. return
        return bitrate;
    }

    async function getBitratePensieve(prevQuality, buffer, bwPrediction, lastRequested, bitrateArray, nextChunkSize, delay, videoChunkRemain) {
        if (pensieveSession == null) {
            // Lazy loading
            pensieveSession = await ort.InferenceSession.create('./model.onnx');
            console.log('[pensieve] model.onnx loaded.');
        }

        console.log('[pensieve] prevQuality:', prevQuality, ', buffer:', buffer, ', nextChunkSize:', nextChunkSize, ', delay:', delay)

        if (prevQuality == null || isNaN(prevQuality)) {
            console.error("[pensieve] prevQuality is NaN or undefined! Changing it to 1.");
            prevQuality = 1;
        }

        // Roll
        for (let i = 0; i < S_INFO; i++) {
            pensieveState[i].push(pensieveState[i].shift());
        }
        
        // Set value
        pensieveState[0][S_LEN - 1] = bitrateArray[prevQuality] / Math.max(...bitrateArray);
        pensieveState[1][S_LEN - 1] = buffer / 10.0;
        pensieveState[2][S_LEN - 1] = nextChunkSize[prevQuality] / delay / 1000.0;
        pensieveState[3][S_LEN - 1] = delay / 1000.0 / 10.0;
        for (let i = 0; i < nextChunkSize.length; i++) {
            pensieveState[4][i] = nextChunkSize[i] / 1000.0 / 1000.0;
        }
        pensieveState[5][S_LEN - 1] = Math.min(videoChunkRemain, TOTAL_VIDEO_CHUNKS) / TOTAL_VIDEO_CHUNKS;
        console.log('[pensieve] pensieveState:', pensieveState);

        // Prepare
        let dataArray = Float32Array.from([].concat(...pensieveState));
        let dataTensor = new ort.Tensor('float32', dataArray, [1, S_INFO, S_LEN]);
        let feeds = { state: dataTensor }

        // Run
        let result = await pensieveSession.run(feeds);
        let probability = result.probability.data;
        console.log('[pensieve] probability:', probability);

        function weightedChoice(array, weights) {
            let s = weights.reduce((a, e) => a + e);
            let r = Math.random() * s;
            return array.find((e, i) => (r -= weights[i]) < 0);
        }

        // Choice
        let bitrate = weightedChoice(Array.from({length: bitrateArray.length}, (x, i) => i), probability);
        console.log('[pensieve] bitrate:', bitrate);

        if (bitrate == null || isNaN(bitrate)) {
            console.error(`[pensieve] bitrate is NaN or undefined! Changing it to ${prevQuality}.`);
            bitrate = prevQuality;
        }

        // 5. return
        return bitrate;
    }

    function predict_throughput(lastRequested, lastQuality, lastHTTPRequest) {
        var self = this,
        bandwidthEst = 0,
        lastDownloadTime,
        lastThroughput,
        lastChunkSize,
        tmpIndex,
        tmpSum = 0,
        tmpDownloadTime = 0;
        // First, log last download time and throughput
        if (lastHTTPRequest && lastRequested >= 0) {
            lastDownloadTime = (lastHTTPRequest._tfinish.getTime() - lastHTTPRequest.tresponse.getTime()) / 1000; //seconds
            if (lastDownloadTime <0.1) {
            lastDownloadTime = 0.1;
            }
            lastChunkSize = last_chunk_size(lastHTTPRequest);
            //lastChunkSize = self.vbr.getChunkSize(lastRequested, lastQuality);
            lastThroughput = lastChunkSize*8/lastDownloadTime/1000;
            // Log last chunk
            pastThroughput[lastRequested] = lastThroughput;
            pastDownloadTime[lastRequested] = lastDownloadTime;
        }
        // Next, predict future bandwidth
        if (pastThroughput.length === 0) {
            return 0;
        } else {
            tmpIndex = Math.max(0, lastRequested + 1 - horizon);
            tmpSum = 0;
            tmpDownloadTime = 0;
            for (var i = tmpIndex; i<= lastRequested; i++) {
            tmpSum = tmpSum + pastDownloadTime[i]/pastThroughput[i];
            tmpDownloadTime = tmpDownloadTime + pastDownloadTime[i];
            }
            bandwidthEst = tmpDownloadTime/tmpSum;
            bandwidthEstLog[lastRequested] = bandwidthEst;
            return bandwidthEst;
        }   
    }

    function setConfig(config) {
        if (!config) return;

        if (config.abrRulesCollection) {
            abrRulesCollection = config.abrRulesCollection;
        }
        if (config.rulesController) {
            rulesController = config.rulesController;
        }
        if (config.streamController) {
            streamController = config.streamController;
        }
    }

    function getBitrateBB(bLevel) {
        var self = this,
        tmpBitrate = 0,
        tmpQuality = 0;
        if (bLevel <= reservoir) {
            tmpBitrate = bitrateArray[0];
        } else if (bLevel > reservoir + cushion) {
            tmpBitrate = bitrateArray[8];
        } else {
            tmpBitrate = bitrateArray[0] + (bitrateArray[8] - bitrateArray[0])*(bLevel - reservoir)/cushion;
        }
        
        // findout matching quality level
        for (var i = 9; i>=0; i--) {
            if (tmpBitrate >= bitrateArray[i]) {
                tmpQuality = i;
                break;
            }
            tmpQuality = i;
        }
        //return 9;
        return tmpQuality;
        // return 0;
    }

    function getBitrateRB(bandwidth) {
        var self = this,
        tmpBitrate = 0,
        tmpQuality = 0;
        
        tmpBitrate = bandwidth*p_rb;
        
        // findout matching quality level
        for (var i = 9; i>=0; i--) {
            if (tmpBitrate >= bitrateArray[i]) {
                tmpQuality = i;
                break;
            }
            tmpQuality = i;
        }
        return tmpQuality;  
        // return 0;
    }

    function getTopQualityIndexFor(type, id) {
        var idx;
        topQualities[id] = topQualities[id] || {};

        if (!topQualities[id].hasOwnProperty(type)) {
            topQualities[id][type] = 0;
        }

        idx = checkMaxBitrate(topQualities[id][type], type);
        idx = checkMaxRepresentationRatio(idx, type, topQualities[id][type]);
        idx = checkPortalSize(idx, type);
        return idx;
    }

    /**
     * @param {string} type
     * @returns {number} A value of the initial bitrate, kbps
     * @memberof AbrController#
     */
    function getInitialBitrateFor(type) {

        let savedBitrate = domStorage.getSavedBitrateSettings(type);

        if (!bitrateDict.hasOwnProperty(type)) {
            if (ratioDict.hasOwnProperty(type)) {
                let manifest = manifestModel.getValue();
                let representation = dashManifestModel.getAdaptationForType(manifest, 0, type).Representation;

                if (Array.isArray(representation)) {
                    let repIdx = Math.max(Math.round(representation.length * ratioDict[type]) - 1, 0);
                    bitrateDict[type] = representation[repIdx].bandwidth;
                } else {
                    bitrateDict[type] = 0;
                }
            } else if (!isNaN(savedBitrate)) {
                bitrateDict[type] = savedBitrate;
            } else {
                bitrateDict[type] = (type === 'video') ? DEFAULT_VIDEO_BITRATE : DEFAULT_AUDIO_BITRATE;
            }
        }

        return bitrateDict[type];
    }

    /**
     * @param {string} type
     * @param {number} value A value of the initial bitrate, kbps
     * @memberof AbrController#
     */
    function setInitialBitrateFor(type, value) {
        bitrateDict[type] = value;
    }

    function getInitialRepresentationRatioFor(type) {
        if (!ratioDict.hasOwnProperty(type)) {
            return null;
        }

        return ratioDict[type];
    }

    function setInitialRepresentationRatioFor(type, value) {
        ratioDict[type] = value;
    }

    function getMaxAllowedBitrateFor(type) {
        if (bitrateDict.hasOwnProperty('max') && bitrateDict.max.hasOwnProperty(type)) {
            return bitrateDict.max[type];
        }
        return NaN;
    }

    //TODO  change bitrateDict structure to hold one object for video and audio with initial and max values internal.
    // This means you need to update all the logic around initial bitrate DOMStorage, RebController etc...
    function setMaxAllowedBitrateFor(type, value) {
        bitrateDict.max = bitrateDict.max || {};
        bitrateDict.max[type] = value;
    }

    function getMaxAllowedRepresentationRatioFor(type) {
        if (ratioDict.hasOwnProperty('max') && ratioDict.max.hasOwnProperty(type)) {
            return ratioDict.max[type];
        }
        return 1;
    }

    function setMaxAllowedRepresentationRatioFor(type, value) {
        ratioDict.max = ratioDict.max || {};
        ratioDict.max[type] = value;
    }

    function getAutoSwitchBitrateFor(type) {
        return autoSwitchBitrate[type];
    }

    function setAutoSwitchBitrateFor(type, value) {
        autoSwitchBitrate[type] = value;
    }

    function getLimitBitrateByPortal() {
        return limitBitrateByPortal;
    }

    function setLimitBitrateByPortal(value) {
        limitBitrateByPortal = value;
    }

    function getUsePixelRatioInLimitBitrateByPortal() {
        return usePixelRatioInLimitBitrateByPortal;
    }

    function setUsePixelRatioInLimitBitrateByPortal(value) {
        usePixelRatioInLimitBitrateByPortal = value;
    }

    async function nextChunkQuality(buffer, lastRequested, lastQuality, rebuffer) {
        const metrics = metricsModel.getReadOnlyMetricsFor('video');
        //console.log("ORIG THROUGH: " + getAverageThroughput("video"));
        //var lastHTTPRequest = dashMetrics.getHttpRequests(metricsModel.getReadOnlyMetricsFor('video'))[lastRequested];
        var lastHTTPRequest = dashMetrics.getCurrentHttpRequest(metrics);
        var bandwidthEst = predict_throughput(lastRequested, lastQuality, lastHTTPRequest);
        switch(abrAlgo) {
            case 2:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'BB', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return getBitrateBB(buffer);
            case 3:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'RB', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return getBitrateRB(bandwidthEst);
            case 4:
                var quality = 2;
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText != "REFRESH" ) {
                            quality = parseInt(xhr.responseText, 10);
                        } else {
                            document.location.reload(true);
                        }
                    }
                }
                var bufferLevelAdjusted = buffer-0.15-0.4-4;
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                console.log("QUALITY RETURNED IS: " + quality);
                return quality;
            case 5:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'Festive', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                var bufferLevelAdjusted = buffer-0.15-0.4-4;
                return getBitrateFestive(lastQuality, bufferLevelAdjusted, bandwidthEst, lastRequested, bitrateArray);
            case 6:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'Bola', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return -1;
            case 7:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                console.log('[pensieve] next_chunk_size', next_chunk_size(lastRequested+1, lastQuality));
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'LocalPensieve', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                var bufferLevelAdjusted = buffer-0.15-0.4-4;
                console.log('[pensieve] Using local pensieve.')
                return await getBitratePensieve(lastQuality, buffer, bandwidthEst, lastRequested, bitrateArray, data['nextChunkSize'], data['lastChunkFinishTime'] - data['lastChunkStartTime'], 48 - lastRequested);
            default:
                // defaults to lowest quality always
                var xhr = new XMLHttpRequest();
                xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'Fixed Quality (0)', 'lastquality': 0, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return 0;
        }
    }

    async function getPlaybackQuality(streamProcessor, completedCallback, buffer=0, rebuffer=0) {
        const type = streamProcessor.getType();
        const streamInfo = streamProcessor.getStreamInfo();
        const streamId = streamInfo.id;

        const callback = function (res) {

            const topQualityIdx = getTopQualityIndexFor(type, streamId);

            let newQuality = res.value;
            if (newQuality < 0) {
                newQuality = 0;
            }
            if (newQuality > topQualityIdx) {
                newQuality = topQualityIdx;
            }

            const oldQuality = getQualityFor(type, streamInfo);
            if (newQuality !== oldQuality && (abandonmentStateDict[type].state === ALLOW_LOAD || newQuality > oldQuality)) {
                setConfidenceFor(type, streamId, res.confidence);
                changeQuality(type, streamInfo, oldQuality, newQuality, res.reason);
            }
            if (completedCallback) {
                completedCallback();
            }
        };

        //log("ABR enabled? (" + autoSwitchBitrate + ")");
        if (!getAutoSwitchBitrateFor(type)) {
            if (completedCallback) {
                completedCallback();
            }
        } else {
            const rules = abrRulesCollection.getRules(ABRRulesCollection.QUALITY_SWITCH_RULES);
            rulesController.applyRules(rules, streamProcessor, callback, getQualityFor(type, streamInfo), async function (currentValue, newValue) {
                currentValue = currentValue === SwitchRequest.NO_CHANGE ? 0 : currentValue;
                if ( abrAlgo == 0 ) { // use the default return value
                    const metrics = metricsModel.getReadOnlyMetricsFor('video');
                    var lastHTTPRequest = dashMetrics.getCurrentHttpRequest(metrics);
                    var bandwidthEst = predict_throughput(lastRequested, lastQuality, lastHTTPRequest);
                    // defaults to lowest quality always
                    var xhr = new XMLHttpRequest();
                    xhr.open("POST", `http://${HOST_IP}:${PORT}`, false);
                    xhr.onreadystatechange = function() {
                        if ( xhr.readyState == 4 && xhr.status == 200 ) {
                            console.log("GOT RESPONSE:" + xhr.responseText + "---");
                            if ( xhr.responseText == "REFRESH" ) {
                                document.location.reload(true);
                            }
                        }
                    }
                    var bufferLevelAdjusted = buffer-0.15-0.4-4;
                    var data = {'nextChunkSize': next_chunk_size(lastRequested+1, lastQuality), 'Type': 'Default', 'lastquality': 0, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                    xhr.send(JSON.stringify(data));
                    return Math.max(currentValue, newValue);
                }
                lastQuality = await nextChunkQuality(buffer, lastRequested, lastQuality, rebuffer);
                lastRequested = lastRequested + 1;
                if ( abrAlgo == 6 ) {
                    lastQuality = Math.max(currentValue, newValue);
                    return Math.max(currentValue, newValue);
                }
                newValue = lastQuality;
                return lastQuality;
            });
        }
    }

    function setAbrAlgorithm(algo) {
        abrAlgo = algo;
    }

    function setPlaybackQuality(type, streamInfo, newQuality, reason) {
        const id = streamInfo.id;
        const oldQuality = getQualityFor(type, streamInfo);
        const isInt = newQuality !== null && !isNaN(newQuality) && (newQuality % 1 === 0);

        if (!isInt) throw new Error('argument is not an integer');

        if (newQuality !== oldQuality && newQuality >= 0 && newQuality <= getTopQualityIndexFor(type, id)) {
            changeQuality(type, streamInfo, oldQuality, newQuality, reason);
        }
    }

    function changeQuality(type, streamInfo, oldQuality, newQuality, reason) {
        setQualityFor(type, streamInfo.id, newQuality);
        eventBus.trigger(Events.QUALITY_CHANGE_REQUESTED, {mediaType: type, streamInfo: streamInfo, oldQuality: oldQuality, newQuality: newQuality, reason: reason});
    }


    function setAbandonmentStateFor(type, state) {
        abandonmentStateDict[type].state = state;
    }

    function getAbandonmentStateFor(type) {
        return abandonmentStateDict[type].state;
    }

    /**
     * @param {MediaInfo} mediaInfo
     * @param {number} bitrate A bitrate value, kbps
     * @returns {number} A quality index <= for the given bitrate
     * @memberof AbrController#
     */
    function getQualityForBitrate(mediaInfo, bitrate) {

        const bitrateList = getBitrateList(mediaInfo);

        if (!bitrateList || bitrateList.length === 0) {
            return QUALITY_DEFAULT;
        }

        for (let i = bitrateList.length - 1; i >= 0; i--) {
            const bitrateInfo = bitrateList[i];
            if (bitrate * 1000 >= bitrateInfo.bitrate) {
                return i;
            }
        }
        return 0;
    }

    /**
     * @param {MediaInfo} mediaInfo
     * @returns {Array|null} A list of {@link BitrateInfo} objects
     * @memberof AbrController#
     */
    function getBitrateList(mediaInfo) {
        if (!mediaInfo || !mediaInfo.bitrateList) return null;

        var bitrateList = mediaInfo.bitrateList;
        var type = mediaInfo.type;

        var infoList = [];
        var bitrateInfo;

        for (var i = 0, ln = bitrateList.length; i < ln; i++) {
            bitrateInfo = new BitrateInfo();
            bitrateInfo.mediaType = type;
            bitrateInfo.qualityIndex = i;
            bitrateInfo.bitrate = bitrateList[i].bandwidth;
            bitrateInfo.width = bitrateList[i].width;
            bitrateInfo.height = bitrateList[i].height;
            infoList.push(bitrateInfo);
        }

        return infoList;
    }

    function setAverageThroughput(type, value) {
        averageThroughputDict[type] = value;
    }

    function getAverageThroughput(type) {
        return averageThroughputDict[type];
    }

    function updateTopQualityIndex(mediaInfo) {
        var type = mediaInfo.type;
        var streamId = mediaInfo.streamInfo.id;
        var max = mediaInfo.representationCount - 1;

        setTopQualityIndex(type, streamId, max);

        return max;
    }

    function isPlayingAtTopQuality(streamInfo) {
        var isAtTop;
        var streamId = streamInfo.id;
        var audioQuality = getQualityFor('audio', streamInfo);
        var videoQuality = getQualityFor('video', streamInfo);

        isAtTop = (audioQuality === getTopQualityIndexFor('audio', streamId)) &&
            (videoQuality === getTopQualityIndexFor('video', streamId));

        return isAtTop;
    }

    async function reset () {
        eventBus.off(Events.LOADING_PROGRESS, onFragmentLoadProgress, this);
        clearTimeout(abandonmentTimeout);
        abandonmentTimeout = null;
        await setup();
    }

    function getQualityFor(type, streamInfo) {
        var id = streamInfo.id;
        var quality;

        qualityDict[id] = qualityDict[id] || {};

        if (!qualityDict[id].hasOwnProperty(type)) {
            qualityDict[id][type] = QUALITY_DEFAULT;
        }

        quality = qualityDict[id][type];
        return quality;
    }

    function setQualityFor(type, id, value) {
        qualityDict[id] = qualityDict[id] || {};
        qualityDict[id][type] = value;
    }

    function getConfidenceFor(type, id) {
        var confidence;

        confidenceDict[id] = confidenceDict[id] || {};

        if (!confidenceDict[id].hasOwnProperty(type)) {
            confidenceDict[id][type] = 0;
        }

        confidence = confidenceDict[id][type];

        return confidence;
    }

    function setConfidenceFor(type, id, value) {
        confidenceDict[id] = confidenceDict[id] || {};
        confidenceDict[id][type] = value;
    }

    function setTopQualityIndex(type, id, value) {
        topQualities[id] = topQualities[id] || {};
        topQualities[id][type] = value;
    }

    function checkMaxBitrate(idx, type) {
        var maxBitrate = getMaxAllowedBitrateFor(type);
        if (isNaN(maxBitrate) || !streamProcessorDict[type]) {
            return idx;
        }
        var maxIdx = getQualityForBitrate(streamProcessorDict[type].getMediaInfo(), maxBitrate);
        return Math.min (idx , maxIdx);
    }

    function checkMaxRepresentationRatio(idx, type, maxIdx) {
        var maxRepresentationRatio = getMaxAllowedRepresentationRatioFor(type);
        if (isNaN(maxRepresentationRatio) || maxRepresentationRatio >= 1 || maxRepresentationRatio < 0) {
            return idx;
        }
        return Math.min( idx , Math.round(maxIdx * maxRepresentationRatio) );
    }

    function checkPortalSize(idx, type) {
        if (type !== 'video' || !limitBitrateByPortal || !streamProcessorDict[type]) {
            return idx;
        }

        var hasPixelRatio = usePixelRatioInLimitBitrateByPortal && window.hasOwnProperty('devicePixelRatio');
        var pixelRatio = hasPixelRatio ? window.devicePixelRatio : 1;
        var element = videoModel.getElement();
        var elementWidth = element.clientWidth * pixelRatio;
        var elementHeight = element.clientHeight * pixelRatio;
        var manifest = manifestModel.getValue();
        var representation = dashManifestModel.getAdaptationForType(manifest, 0, type).Representation;
        var newIdx = idx;

        if (elementWidth > 0 && elementHeight > 0) {
            while (
                newIdx > 0 &&
                representation[newIdx] &&
                elementWidth < representation[newIdx].width &&
                elementWidth - representation[newIdx - 1].width < representation[newIdx].width - elementWidth
            ) {
                newIdx = newIdx - 1;
            }

            if (representation.length - 2 >= newIdx && representation[newIdx].width === representation[newIdx + 1].width) {
                newIdx = Math.min(idx, newIdx + 1);
            }
        }

        return newIdx;
    }

    function onFragmentLoadProgress(e) {
        const type = e.request.mediaType;
        if (getAutoSwitchBitrateFor(type)) {

            const rules = abrRulesCollection.getRules(ABRRulesCollection.ABANDON_FRAGMENT_RULES);
            const scheduleController = streamProcessorDict[type].getScheduleController();
            if (!scheduleController) return;// There may be a fragment load in progress when we switch periods and recreated some controllers.

            const callback = function (switchRequest) {
                if (switchRequest.confidence === SwitchRequest.STRONG &&
                    switchRequest.value < getQualityFor(type, streamController.getActiveStreamInfo())) {

                    const fragmentModel = scheduleController.getFragmentModel();
                    const request = fragmentModel.getRequests({state: FragmentModel.FRAGMENT_MODEL_LOADING, index: e.request.index})[0];
                    if (request) {
                        //TODO Check if we should abort or if better to finish download. check bytesLoaded/Total
                        fragmentModel.abortRequests();
                        setAbandonmentStateFor(type, ABANDON_LOAD);
                        setPlaybackQuality(type, streamController.getActiveStreamInfo(), switchRequest.value, switchRequest.reason);
                        eventBus.trigger(Events.FRAGMENT_LOADING_ABANDONED, {streamProcessor: streamProcessorDict[type], request: request, mediaType: type});

                        clearTimeout(abandonmentTimeout);
                        abandonmentTimeout = setTimeout(() => {
                            setAbandonmentStateFor(type, ALLOW_LOAD);
                            abandonmentTimeout = null;
                        }, mediaPlayerModel.getAbandonLoadTimeout());
                    }
                }
            };

            rulesController.applyRules(rules, streamProcessorDict[type], callback, e, async function (currentValue, newValue) {
                return newValue;
            });
        }
    }

    instance = {
        isPlayingAtTopQuality: isPlayingAtTopQuality,
        updateTopQualityIndex: updateTopQualityIndex,
        getAverageThroughput: getAverageThroughput,
        getBitrateList: getBitrateList,
        getQualityForBitrate: getQualityForBitrate,
        getMaxAllowedBitrateFor: getMaxAllowedBitrateFor,
        setMaxAllowedBitrateFor: setMaxAllowedBitrateFor,
        getMaxAllowedRepresentationRatioFor: getMaxAllowedRepresentationRatioFor,
        setMaxAllowedRepresentationRatioFor: setMaxAllowedRepresentationRatioFor,
        getInitialBitrateFor: getInitialBitrateFor,
        setInitialBitrateFor: setInitialBitrateFor,
        getInitialRepresentationRatioFor: getInitialRepresentationRatioFor,
        setInitialRepresentationRatioFor: setInitialRepresentationRatioFor,
        setAutoSwitchBitrateFor: setAutoSwitchBitrateFor,
        getAutoSwitchBitrateFor: getAutoSwitchBitrateFor,
        setLimitBitrateByPortal: setLimitBitrateByPortal,
        getLimitBitrateByPortal: getLimitBitrateByPortal,
        getUsePixelRatioInLimitBitrateByPortal: getUsePixelRatioInLimitBitrateByPortal,
        setUsePixelRatioInLimitBitrateByPortal: setUsePixelRatioInLimitBitrateByPortal,
        getConfidenceFor: getConfidenceFor,
        getQualityFor: getQualityFor,
        getAbandonmentStateFor: getAbandonmentStateFor,
        setAbandonmentStateFor: setAbandonmentStateFor,
        setPlaybackQuality: setPlaybackQuality,
        setAbrAlgorithm: setAbrAlgorithm,
        getPlaybackQuality: getPlaybackQuality,
        setAverageThroughput: setAverageThroughput,
        getTopQualityIndexFor: getTopQualityIndexFor,
        initialize: initialize,
        setConfig: setConfig,
        reset: reset
    };

    setup();

    return instance;
}

AbrController.__dashjs_factory_name = 'AbrController';
let factory = FactoryMaker.getSingletonFactory(AbrController);
factory.ABANDON_LOAD = ABANDON_LOAD;
factory.QUALITY_DEFAULT = QUALITY_DEFAULT;
export default factory;
