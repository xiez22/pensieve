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
import UTCTiming from '../dash/vo/UTCTiming';
import PlaybackController from './controllers/PlaybackController';
import StreamController from './controllers/StreamController';
import MediaController from './controllers/MediaController';
import ManifestLoader from './ManifestLoader';
import LiveEdgeFinder from './utils/LiveEdgeFinder';
import ErrorHandler from './utils/ErrorHandler';
import Capabilities from './utils/Capabilities';
import TextTracks from './TextTracks';
import SourceBufferController from './controllers/SourceBufferController';
import RequestModifier from './utils/RequestModifier';
import TextSourceBuffer from './TextSourceBuffer';
import URIQueryAndFragmentModel from './models/URIQueryAndFragmentModel';
import ManifestModel from './models/ManifestModel';
import MediaPlayerModel from './models/MediaPlayerModel';
import MetricsModel from './models/MetricsModel';
import AbrController from './controllers/AbrController';
import TimeSyncController from './controllers/TimeSyncController';
import ABRRulesCollection from './rules/abr/ABRRulesCollection';
import VideoModel from './models/VideoModel';
import RulesController from './rules/RulesController';
import SynchronizationRulesCollection from './rules/synchronization/SynchronizationRulesCollection';
import MediaSourceController from './controllers/MediaSourceController';
import BaseURLController from './controllers/BaseURLController';
import Debug from './../core/Debug';
import EventBus from './../core/EventBus';
import Events from './../core/events/Events';
import MediaPlayerEvents from './MediaPlayerEvents';
import FactoryMaker from '../core/FactoryMaker';
import {getVersionString} from './../core/Version';

//Dash
import DashAdapter from '../dash/DashAdapter';
import DashParser from '../dash/parser/DashParser';
import DashManifestModel from '../dash/models/DashManifestModel';
import DashMetrics from '../dash/DashMetrics';
import TimelineConverter from '../dash/utils/TimelineConverter';

/**
 * @module MediaPlayer
 * @description The MediaPlayer is the primary dash.js Module and a Facade to build your player around.
 * It will allow you access to all the important dash.js properties/methods via the public API and all the
 * events to build a robust DASH media player.
 */
function MediaPlayer() {

    const PLAYBACK_NOT_INITIALIZED_ERROR = 'You must first call play() to init playback before calling this method';
    const ELEMENT_NOT_ATTACHED_ERROR = 'You must first call attachView() to set the video element before calling this method';
    const SOURCE_NOT_ATTACHED_ERROR = 'You must first call attachSource() with a valid source before calling this method';
    const MEDIA_PLAYER_NOT_INITIALIZED_ERROR = 'MediaPlayer not initialized!';

    let context = this.context;
    let eventBus = EventBus(context).getInstance();
    let debug = Debug(context).getInstance();
    let log = debug.log;

    let instance,
        source,
        protectionData,
        mediaPlayerInitialized,
        playbackInitialized,
        autoPlay,
        abrController,
        mediaController,
        protectionController,
        metricsReportingController,
        adapter,
        metricsModel,
        mediaPlayerModel,
        errHandler,
        capabilities,
        streamController,
        rulesController,
        playbackController,
        dashMetrics,
        dashManifestModel,
        videoModel,
        textSourceBuffer;

    function setup() {
        mediaPlayerInitialized = false;
        playbackInitialized = false;
        autoPlay = true;
        protectionController = null;
        protectionData = null;
        adapter = null;
        Events.extend(MediaPlayerEvents);
        mediaPlayerModel = MediaPlayerModel(context).getInstance();
    }

    /**
     * Upon creating the MediaPlayer you must call initialize before you call anything else.
     * There is one exception to this rule. It is crucial to call {@link module:MediaPlayer#extend extend()}
     * with all your extensions prior to calling initialize.
     *
     * ALL arguments are optional and there are individual methods to set each argument later on.
     * The args in this method are just for convenience and should only be used for a simple player setup.
     *
     * @param {HTML5MediaElement=} view - Optional arg to set the video element. {@link module:MediaPlayer#attachView attachView()}
     * @param {string=} source - Optional arg to set the media source. {@link module:MediaPlayer#attachSource attachSource()}
     * @param {boolean=} AutoPlay - Optional arg to set auto play. {@link module:MediaPlayer#setAutoPlay setAutoPlay()}
     * @see {@link module:MediaPlayer#attachView attachView()}
     * @see {@link module:MediaPlayer#attachSource attachSource()}
     * @see {@link module:MediaPlayer#setAutoPlay setAutoPlay()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function initialize(view, source, AutoPlay) {
        capabilities = Capabilities(context).getInstance();
        errHandler = ErrorHandler(context).getInstance();

        if (!capabilities.supportsMediaSource()) {
            errHandler.capabilityError('mediasource');
            return;
        }

        if (mediaPlayerInitialized) return;
        mediaPlayerInitialized = true;

        abrController = AbrController(context).getInstance();

        playbackController = PlaybackController(context).getInstance();
        mediaController = MediaController(context).getInstance();
        mediaController.initialize();
        dashManifestModel = DashManifestModel(context).getInstance();
        dashMetrics = DashMetrics(context).getInstance();
        metricsModel = MetricsModel(context).getInstance();
        metricsModel.setConfig({adapter: createAdaptor()});

        restoreDefaultUTCTimingSources();
        setAutoPlay(AutoPlay !== undefined ? AutoPlay : true);

        if (view) {
            attachView(view);
        }

        if (source) {
            attachSource(source);
        }

        log('[dash.js ' + getVersion() + '] ' + 'MediaPlayer has been initialized');
    }

    function isInitialized() {
        return mediaPlayerInitialized;
    }

    function setAbrAlgorithm(abrAlgo) {
        abrController.setAbrAlgorithm(abrAlgo);
    }

    /**
     * The ready state of the MediaPlayer based on both the video element and MPD source being defined.
     *
     * @returns {boolean} The current ready state of the MediaPlayer
     * @see {@link module:MediaPlayer#attachView attachView()}
     * @see {@link module:MediaPlayer#attachSource attachSource()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function isReady() {
        return (!!videoModel && !!source);
    }

    /**
     * The play method initiates playback of the media defined by the {@link module:MediaPlayer#attachSource attachSource()} method.
     * This method will call play on the native Video Element.
     *
     * @see {@link module:MediaPlayer#attachSource attachSource()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function play() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        if (!autoPlay || (isPaused() && playbackInitialized)) {
            playbackController.play();
        }
    }

    /**
     * This method will call pause on the native Video Element.
     *
     * @memberof module:MediaPlayer
     * @instance
     */
    function pause() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        playbackController.pause();
    }

    /**
     * Returns a Boolean that indicates whether the Video Element is paused.
     * @return {boolean}
     * @memberof module:MediaPlayer
     * @instance
     */
    function isPaused() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        return playbackController.isPaused();
    }

    /**
     * Returns a Boolean that indicates whether the media is in the process of seeking to a new position.
     * @return {boolean}
     * @memberof module:MediaPlayer
     * @instance
     */
    function isSeeking() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        return playbackController.isSeeking();
    }

    /**
     * Use this method to set the native Video Element's muted state. Takes a Boolean that determines whether audio is muted. true if the audio is muted and false otherwise.
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setMute(value) {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        getVideoElement().muted = value;
    }

    /**
     * A Boolean that determines whether audio is muted.
     * @returns {boolean}
     * @memberof module:MediaPlayer
     * @instance
     */
    function isMuted() {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        return getVideoElement().muted;
    }

    /**
     * A double indicating the audio volume, from 0.0 (silent) to 1.0 (loudest).
     * @param {number} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setVolume(value) {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        getVideoElement().volume = value;
    }

    /**
     * Returns the current audio volume, from 0.0 (silent) to 1.0 (loudest).
     * @returns {number}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getVolume() {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        return getVideoElement().volume;
    }

    /**
     * The length of the buffer for a given media type, in seconds. Valid media
     * types are "video", "audio" and "fragmentedText". If no type is passed
     * in, then the minimum of video, audio and fragmentedText buffer length is
     * returned. NaN is returned if an invalid type is requested, the
     * presentation does not contain that type, or if no arguments are passed
     * and the presentation does not include any adaption sets of valid media
     * type.
     *
     * @param {string} type - the media type of the buffer
     * @returns {number} The length of the buffer for the given media type, in
     *  seconds, or NaN
     * @memberof module:MediaPlayer
     * @instance
     */
    function getBufferLength(type) {
        const types = ['video', 'audio', 'fragmentedText'];
        if (!type) {
            return types.map(
                t => getTracksFor(t).length > 0 ? getDashMetrics().getCurrentBufferLevel(getMetricsFor(t)) : Number.MAX_VALUE
            ).reduce(
                (p, c) => Math.min(p, c)
            );
        } else {
            if (types.indexOf(type) !== -1) {
                const buffer = getDashMetrics().getCurrentBufferLevel(getMetricsFor(type));
                return buffer ? buffer : NaN;
            } else {
                log('Warning  - getBufferLength requested for invalid type');
                return NaN;
            }
        }
    }

    /**
     * The timeShiftBufferLength (DVR Window), in seconds.
     *
     * @returns {number} The window of allowable play time behind the live point of a live stream.
     * @memberof module:MediaPlayer
     * @instance
     */
    function getDVRWindowSize() {
        var metric = getDVRInfoMetric();
        if (!metric) {
            return 0;
        }
        return metric.manifestInfo.DVRWindowSize;
    }

    /**
     * This method should only be used with a live stream that has a valid timeShiftBufferLength (DVR Window).
     * NOTE - If you do not need the raw offset value (i.e. media analytics, tracking, etc) consider using the {@link module:MediaPlayer#seek seek()} method
     * which will calculate this value for you and set the video element's currentTime property all in one simple call.
     *
     * @param {number} value - A relative time, in seconds, based on the return value of the {@link module:MediaPlayer#duration duration()} method is expected.
     * @returns {number} A value that is relative the available range within the timeShiftBufferLength (DVR Window).
     * @see {@link module:MediaPlayer#seek seek()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getDVRSeekOffset(value) {
        var metric = getDVRInfoMetric();

        if (!metric) {
            return 0;
        }

        var val = metric.range.start + value;

        if (val > metric.range.end) {
            val = metric.range.end;
        }

        return val;
    }

    /**
     * Sets the currentTime property of the attached video element.  If it is a live stream with a
     * timeShiftBufferLength, then the DVR window offset will be automatically calculated.
     *
     * @param {number} value - A relative time, in seconds, based on the return value of the {@link module:MediaPlayer#duration duration()} method is expected
     * @see {@link module:MediaPlayer#getDVRSeekOffset getDVRSeekOffset()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function seek(value) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var s = playbackController.getIsDynamic() ? getDVRSeekOffset(value) : value;
        playbackController.seek(s);
    }


    /**
     * Current time of the playhead, in seconds.
     *
     * If called with no arguments then the returned time value is time elapsed since the start point of the first stream, or if it is a live stream, then the time will be based on the return value of the {@link module:MediaPlayer#duration duration()} method.
     * However if a stream ID is supplied then time is relative to the start of that stream, or is null if there is no such stream id in the manifest.
     *
     * @param {string} streamId - The ID of a stream that the returned playhead time must be relative to the start of. If undefined, then playhead time is relative to the first stream.
     * @returns {number} The current playhead time of the media, or null.
     * @memberof module:MediaPlayer
     * @instance
     */
    function time(streamId) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var t = getVideoElement().currentTime;

        if (streamId !== undefined) {
            t = streamController.getTimeRelativeToStreamId(t, streamId);

        } else if (playbackController.getIsDynamic()) {
            var metric = getDVRInfoMetric();
            t = (metric === null) ? 0 : duration() - (metric.range.end - metric.time);
        }

        return t;
    }

    /**
     * Duration of the media's playback, in seconds.
     *
     * @returns {number} The current duration of the media.
     * @memberof module:MediaPlayer
     * @instance
     */
    function duration() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var d = getVideoElement().duration;

        if (playbackController.getIsDynamic()) {

            var metric = getDVRInfoMetric();
            var range;

            if (!metric) {
                return 0;
            }

            range = metric.range.end - metric.range.start;
            d = range < metric.manifestInfo.DVRWindowSize ? range : metric.manifestInfo.DVRWindowSize;
        }
        return d;
    }

    /**
     * Use this method to get the current playhead time as an absolute value, the time in seconds since midnight UTC, Jan 1 1970.
     * Note - this property only has meaning for live streams. If called before play() has begun, it will return a value of NaN.
     *
     * @returns {number} The current playhead time as UTC timestamp.
     * @memberof module:MediaPlayer
     * @instance
     */
    function timeAsUTC() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        if (time() < 0) {
            return NaN;
        }
        return getAsUTC(time());
    }

    /**
     * Use this method to get the current duration as an absolute value, the time in seconds since midnight UTC, Jan 1 1970.
     * Note - this property only has meaning for live streams.
     *
     * @returns {number} The current duration as UTC timestamp.
     * @memberof module:MediaPlayer
     * @instance
     */
    function durationAsUTC() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        return getAsUTC(duration());
    }

    /**
     * A utility methods which converts UTC timestamp value into a valid time and date string.
     *
     * @param {number} time - UTC timestamp to be converted into date and time.
     * @param {string} locales - a region identifier (i.e. en_US).
     * @param {boolean} hour12 - 12 vs 24 hour. Set to true for 12 hour time formatting.
     * @returns {string} A formatted time and date string.
     * @memberof module:MediaPlayer
     * @instance
     */
    function formatUTC(time, locales, hour12) {
        var dt = new Date(time * 1000);
        var d = dt.toLocaleDateString(locales);
        var t = dt.toLocaleTimeString(locales, {hour12: hour12});
        return t + ' ' + d;
    }

    /**
     * A utility method which converts seconds into TimeCode (i.e. 300 --> 05:00).
     *
     * @param {number} value - A number in seconds to be converted into a formatted time code.
     * @returns {string} A formatted time code string.
     * @memberof module:MediaPlayer
     * @instance
     */
    function convertToTimeCode(value) {
        value = Math.max(value, 0);

        var h = Math.floor(value / 3600);
        var m = Math.floor((value % 3600) / 60);
        var s = Math.floor((value % 3600) % 60);
        return (h === 0 ? '' : (h < 10 ? '0' + h.toString() + ':' : h.toString() + ':')) + (m < 10 ? '0' + m.toString() : m.toString()) + ':' + (s < 10 ? '0' + s.toString() : s.toString());
    }

    /**
     * This method should be used to extend or replace internal dash.js objects.
     * There are two ways to extend dash.js (determined by the override argument):
     * <ol>
     * <li>If you set override to true any public method or property in your custom object will
     * override the dash.js parent object's property(ies) and will be used instead but the
     * dash.js parent module will still be created.</li>
     *
     * <li>If you set override to false your object will completely replace the dash.js object.
     * (Note: This is how it was in 1.x of Dash.js with Dijon).</li>
     * </ol>
     * <b>When you extend you get access to this.context, this.factory and this.parent to operate with in your custom object.</b>
     * <ul>
     * <li><b>this.context</b> - can be used to pass context for singleton access.</li>
     * <li><b>this.factory</b> - can be used to call factory.getSingletonInstance().</li>
     * <li><b>this.parent</b> - is the reference of the parent object to call other public methods. (this.parent is excluded if you extend with override set to false or option 2)</li>
     * </ul>
     * <b>You must call extend before you call initialize</b>
     * @see {@link module:MediaPlayer#initialize initialize()}
     * @param {string} parentNameString - name of parent module
     * @param {Object} childInstance - overriding object
     * @param {boolean} override - replace only some methods (true) or the whole object (false)
     * @memberof module:MediaPlayer
     * @instance
     */
    function extend(parentNameString, childInstance, override) {
        FactoryMaker.extend(parentNameString, childInstance, override, context);
    }

    /**
     * Use the on method to listen for public events found in MediaPlayer.events. {@link MediaPlayerEvents}
     *
     * @param {string} type - {@link MediaPlayerEvents}
     * @param {Function} listener - callback method when the event fires.
     * @param {Object} scope - context of the listener so it can be removed properly.
     * @memberof module:MediaPlayer
     * @instance
     */
    function on(type, listener, scope) {
        eventBus.on(type, listener, scope);
    }

    /**
     * Use the off method to remove listeners for public events found in MediaPlayer.events. {@link MediaPlayerEvents}
     *
     * @param {string} type - {@link MediaPlayerEvents}
     * @param {Function} listener - callback method when the event fires.
     * @param {Object} scope - context of the listener so it can be removed properly.
     * @memberof module:MediaPlayer
     * @instance
     */
    function off(type, listener, scope) {
        eventBus.off(type, listener, scope);
    }

    /**
     * Current version of Dash.js
     * @returns {string} the current dash.js version string.
     * @memberof module:MediaPlayer
     * @instance
     */
    function getVersion() {
        return getVersionString();
    }

    /**
     * Use this method to access the dash.js logging class.
     *
     * @returns {Debug}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getDebug() {
        return debug;
    }

    /**
     * @deprecated Since version 2.1.0.  <b>Instead use:</b>
     * <ul>
     * <li>{@link module:MediaPlayer#getVideoElement getVideoElement()}</li>
     * <li>{@link module:MediaPlayer#getSource getSource()}</li>
     * <li>{@link module:MediaPlayer#getVideoContainer getVideoContainer()}</li>
     * <li>{@link module:MediaPlayer#getTTMLRenderingDiv getTTMLRenderingDiv()}</li>
     * </ul>
     *
     * @returns {VideoModel}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getVideoModel() {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        return videoModel;
    }

    /**
     * <p>Changing this value will lower or increase live stream latency.  The detected segment duration will be multiplied by this value
     * to define a time in seconds to delay a live stream from the live edge.</p>
     * <p>Lowering this value will lower latency but may decrease the player's ability to build a stable buffer.</p>
     *
     * @param {number} value - Represents how many segment durations to delay the live stream.
     * @default 4
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#useSuggestedPresentationDelay useSuggestedPresentationDelay()}
     * @instance
     */
    function setLiveDelayFragmentCount(value) {
        mediaPlayerModel.setLiveDelayFragmentCount(value);
    }

    /**
     * <p>Equivalent in seconds of setLiveDelayFragmentCount</p>
     * <p>Lowering this value will lower latency but may decrease the player's ability to build a stable buffer.</p>
     * <p>This value should be less than the manifest duration by a couple of segment durations to avoid playback issues</p>
     * <p>If set, this parameter will take precedence over setLiveDelayFragmentCount and manifest info</p>
     *
     * @param {number} value - Represents how many seconds to delay the live stream.
     * @default undefined
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#useSuggestedPresentationDelay useSuggestedPresentationDelay()}
     * @instance
     */
    function setLiveDelay(value) {
        mediaPlayerModel.setLiveDelay(value);
    }

    /**
     * <p>Set to true if you would like to override the default live delay and honor the SuggestedPresentationDelay attribute in by the manifest.</p>
     * @param {boolean} value
     * @default false
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#setLiveDelayFragmentCount setLiveDelayFragmentCount()}
     * @instance
     */
    function useSuggestedPresentationDelay(value) {
        mediaPlayerModel.setUseSuggestedPresentationDelay(value);
    }

    /**
     * Set to false if you would like to disable the last known bit rate from being stored during playback and used
     * to set the initial bit rate for subsequent playback within the expiration window.
     *
     * The default expiration is one hour, defined in milliseconds. If expired, the default initial bit rate (closest to 1000 kbps) will be used
     * for that session and a new bit rate will be stored during that session.
     *
     * @param {boolean} enable - Will toggle if feature is enabled. True to enable, False to disable.
     * @param {number=} ttl - (Optional) A value defined in milliseconds representing how long to cache the bit rate for. Time to live.
     * @default enable = True, ttl = 360000 (1 hour)
     * @memberof module:MediaPlayer
     * @instance
     *
     */
    function enableLastBitrateCaching(enable, ttl) {
        mediaPlayerModel.setLastBitrateCachingInfo(enable, ttl);
    }

    /**
     * Set to false if you would like to disable the last known lang for audio (or camera angle for video) from being stored during playback and used
     * to set the initial settings for subsequent playback within the expiration window.
     *
     * The default expiration is one hour, defined in milliseconds. If expired, the default settings will be used
     * for that session and a new settings will be stored during that session.
     *
     * @param {boolean} enable - Will toggle if feature is enabled. True to enable, False to disable.
     * @param {number=} [ttl] - (Optional) A value defined in milliseconds representing how long to cache the settings for. Time to live.
     * @default enable = True, ttl = 360000 (1 hour)
     * @memberof module:MediaPlayer
     * @instance
     *
     */
    function enableLastMediaSettingsCaching(enable, ttl) {
        mediaPlayerModel.setLastMediaSettingsCachingInfo(enable, ttl);
    }

    /**
     * When switching multi-bitrate content (auto or manual mode) this property specifies the maximum bitrate allowed.
     * If you set this property to a value lower than that currently playing, the switching engine will switch down to
     * satisfy this requirement. If you set it to a value that is lower than the lowest bitrate, it will still play
     * that lowest bitrate.
     *
     * You can set or remove this bitrate cap at anytime before or during playback.  To clear this setting you must use the API
     * and set the value param to NaN.
     *
     * This feature is typically used to reserve higher bitrates for playback only when the player is in large or full-screen format.
     *
     * @param {string} type - 'video' or 'audio' are the type options.
     * @param {number} value - Value in kbps representing the maximum bitrate allowed.
     * @memberof module:MediaPlayer
     * @instance
     */
    function setMaxAllowedBitrateFor(type, value) {
        abrController.setMaxAllowedBitrateFor(type, value);
    }

    /**
     * @param {string} type - 'video' or 'audio' are the type options.
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#setMaxAllowedBitrateFor setMaxAllowedBitrateFor()}
     * @instance
     */
    function getMaxAllowedBitrateFor(type) {
        return abrController.getMaxAllowedBitrateFor(type);
    }

    /**
     * When switching multi-bitrate content (auto or manual mode) this property specifies the maximum representation allowed,
     * as a proportion of the size of the representation set.
     *
     * You can set or remove this cap at anytime before or during playback. To clear this setting you must use the API
     * and set the value param to NaN.
     *
     * If both this and maxAllowedBitrate are defined, maxAllowedBitrate is evaluated first, then maxAllowedRepresentation,
     * i.e. the lowest value from executing these rules is used.
     *
     * This feature is typically used to reserve higher representations for playback only when connected over a fast connection.
     *
     * @param {string} type - 'video' or 'audio' are the type options.
     * @param {number} value - number between 0 and 1, where 1 is allow all representations, and 0 is allow only the lowest.
     * @memberof module:MediaPlayer
     * @instance
     */
    function setMaxAllowedRepresentationRatioFor(type, value) {
        abrController.setMaxAllowedRepresentationRatioFor(type, value);
    }

    /**
     * @param {string} type - 'video' or 'audio' are the type options.
     * @returns {number} The current representation ratio cap.
     * @memberof module:MediaPlayer
     * @see {@link MediaPlayer#setMaxAllowedRepresentationRatioFor setMaxAllowedRepresentationRatioFor()}
     * @instance
     */
    function getMaxAllowedRepresentationRatioFor(type) {
        return abrController.getMaxAllowedRepresentationRatioFor(type);
    }

    /**
     * <p>Set to false to prevent stream from auto-playing when the view is attached.</p>
     *
     * @param {boolean} value
     * @default true
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#attachView attachView()}
     * @instance
     *
     */
    function setAutoPlay(value) {
        autoPlay = value;
    }

    /**
     * @returns {boolean} The current autoPlay state.
     * @memberof module:MediaPlayer
     * @instance
     */
    function getAutoPlay() {
        return autoPlay;
    }

    /**
     * Set to true if you would like dash.js to keep downloading fragments in the background
     * when the video element is paused.
     *
     * @default true
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setScheduleWhilePaused(value) {
        mediaPlayerModel.setScheduleWhilePaused(value);
    }

    /**
     * Returns a boolean of the current state of ScheduleWhilePaused.
     * @returns {boolean}
     * @see {@link module:MediaPlayer#setScheduleWhilePaused setScheduleWhilePaused()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getScheduleWhilePaused() {
        return mediaPlayerModel.getScheduleWhilePaused();
    }


    /**
     * Returns the DashMetrics.js Module. You use this Module to get access to all the public metrics
     * stored in dash.js
     *
     * @see {@link module:DashMetrics}
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getDashMetrics() {
        return dashMetrics;
    }

    /**
     *
     * @param {string} type
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getMetricsFor(type) {
        return metricsModel.getReadOnlyMetricsFor(type);
    }

    /**
     * @param {string} type
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getQualityFor(type) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        return abrController.getQualityFor(type, streamController.getActiveStreamInfo());
    }

    /**
     * Sets the current quality for media type instead of letting the ABR Heuristics automatically selecting it..
     *
     * @param {string} type
     * @param {number} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setQualityFor(type, value) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        abrController.setPlaybackQuality(type, streamController.getActiveStreamInfo(), value);
    }

    /**
     * @memberof module:MediaPlayer
     * @instance
     */
    function getLimitBitrateByPortal() {
        return abrController.getLimitBitrateByPortal();
    }

    /**
     * Sets whether to limit the representation used based on the size of the playback area
     *
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setLimitBitrateByPortal(value) {
        abrController.setLimitBitrateByPortal(value);
    }

    /**
     * @memberof module:MediaPlayer
     * @instance
     */
    function getUsePixelRatioInLimitBitrateByPortal() {
        return abrController.getUsePixelRatioInLimitBitrateByPortal();
    }

    /**
     * Sets whether to take into account the device's pixel ratio when defining the portal dimensions.
     * Useful on, for example, retina displays.
     *
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     * @default {boolean} false
     */
    function setUsePixelRatioInLimitBitrateByPortal(value) {
        abrController.setUsePixelRatioInLimitBitrateByPortal(value);
    }

    /**
     * Use this method to change the current text track for both external time text files and fragmented text tracks. There is no need to
     * set the track mode on the video object to switch a track when using this method.
     *
     * @param {number} idx - Index of track based on the order of the order the tracks are added Use -1 to disable all tracks. (turn captions off).  Use module:MediaPlayer#dashjs.MediaPlayer.events.TEXT_TRACK_ADDED.
     * @see {@link module:MediaPlayer#dashjs.MediaPlayer.events.TEXT_TRACK_ADDED}
     * @memberof module:MediaPlayer
     * @instance
     */
    function setTextTrack(idx) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        //For external time text file,  the only action needed to change a track is marking the track mode to showing.
        // Fragmented text tracks need the additional step of calling textSourceBuffer.setTextTrack();
        if (textSourceBuffer === undefined) {
            textSourceBuffer = TextSourceBuffer(context).getInstance();
        }

        var tracks = getVideoElement().textTracks;
        var ln = tracks.length;

        for (var i = 0; i < ln; i++) {
            var track = tracks[i];
            var mode = idx === i ? 'showing' : 'hidden';

            if (track.mode !== mode) { //checking that mode is not already set by 3rd Party player frameworks that set mode to prevent event retrigger.
                track.mode = mode;
            }
        }

        textSourceBuffer.setTextTrack();
    }

    /**
     * @param {string} type
     * @returns {Array}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getBitrateInfoListFor(type) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var stream = getActiveStream();
        return stream ? stream.getBitrateListFor(type) : [];
    }

    /**
     * Use this method to explicitly set the starting bitrate for audio | video
     *
     * @param {string} type
     * @param {number} value - A value of the initial bitrate, kbps
     * @memberof module:MediaPlayer
     * @instance
     */
    function setInitialBitrateFor(type, value) {
        abrController.setInitialBitrateFor(type, value);
    }

    /**
     * @param {string} type
     * @returns {number} A value of the initial bitrate, kbps
     * @memberof module:MediaPlayer
     * @instance
     */
    function getInitialBitrateFor(type) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR; //abrController.getInitialBitrateFor is overloaded with ratioDict logic that needs manifest force it to not be callable pre play.
        }
        return abrController.getInitialBitrateFor(type);
    }

    /**
     * @param {string} type
     * @param {number} value - A value of the initial Representation Ratio
     * @memberof module:MediaPlayer
     * @instance
     */
    function setInitialRepresentationRatioFor(type, value) {
        abrController.setInitialRepresentationRatioFor(type, value);
    }

    /**
     * @param {string} type
     * @returns {number} A value of the initial Representation Ratio
     * @memberof module:MediaPlayer
     * @instance
     */
    function getInitialRepresentationRatioFor(type) {
        return abrController.getInitialRepresentationRatioFor(type);
    }

    /**
     * This method returns the list of all available streams from a given manifest
     * @param {Object} manifest
     * @returns {Array} list of {@link StreamInfo}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getStreamsFromManifest(manifest) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        return adapter.getStreamsInfo(manifest);
    }

    /**
     * This method returns the list of all available tracks for a given media type
     * @param {string} type
     * @returns {Array} list of {@link MediaInfo}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getTracksFor(type) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        let streamInfo = streamController.getActiveStreamInfo();
        if (!streamInfo) return [];
        return mediaController.getTracksFor(type, streamInfo);
    }

    /**
     * This method returns the list of all available tracks for a given media type and streamInfo from a given manifest
     * @param {string} type
     * @param {Object} manifest
     * @param {Object} streamInfo
     * @returns {Array} list of {@link MediaInfo}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getTracksForTypeFromManifest(type, manifest, streamInfo) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }

        streamInfo = streamInfo || adapter.getStreamsInfo(manifest)[0];

        return streamInfo ? adapter.getAllMediaInfoForType(manifest, streamInfo, type) : [];
    }

    /**
     * @param {string} type
     * @returns {Object|null} {@link MediaInfo}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getCurrentTrackFor(type) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var streamInfo = streamController.getActiveStreamInfo();

        if (!streamInfo) return null;

        return mediaController.getCurrentTrackFor(type, streamInfo);
    }

    /**
     * This method allows to set media settings that will be used to pick the initial track. Format of the settings
     * is following:
     * {lang: langValue,
         *  viewpoint: viewpointValue,
         *  audioChannelConfiguration: audioChannelConfigurationValue,
         *  accessibility: accessibilityValue,
         *  role: roleValue}
     *
     *
     * @param {string} type
     * @param {Object} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setInitialMediaSettingsFor(type, value) {
        mediaController.setInitialSettings(type, value);
    }

    /**
     * This method returns media settings that is used to pick the initial track. Format of the settings
     * is following:
     * {lang: langValue,
         *  viewpoint: viewpointValue,
         *  audioChannelConfiguration: audioChannelConfigurationValue,
         *  accessibility: accessibilityValue,
         *  role: roleValue}
     * @param {string} type
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getInitialMediaSettingsFor(type) {
        return mediaController.getInitialSettings(type);
    }

    /**
     * @param {MediaInfo} track - instance of {@link MediaInfo}
     * @memberof module:MediaPlayer
     * @instance
     */
    function setCurrentTrack(track) {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        mediaController.setTrack(track);
    }

    /**
     * This method returns the current track switch mode.
     *
     * @param {string} type
     * @returns {string} mode
     * @memberof module:MediaPlayer
     * @instance
     */
    function getTrackSwitchModeFor(type) {
        return mediaController.getSwitchMode(type);
    }

    /**
     * This method sets the current track switch mode. Available options are:
     *
     * MediaController.TRACK_SWITCH_MODE_NEVER_REPLACE
     * (used to forbid clearing the buffered data (prior to current playback position) after track switch. Default for video)
     *
     * MediaController.TRACK_SWITCH_MODE_ALWAYS_REPLACE
     * (used to clear the buffered data (prior to current playback position) after track switch. Default for audio)
     *
     * @param {string} type
     * @param {string} mode
     * @memberof module:MediaPlayer
     * @instance
     */
    function setTrackSwitchModeFor(type, mode) {
        mediaController.setSwitchMode(type, mode);
    }

    /**
     * This method sets the selection mode for the initial track. This mode defines how the initial track will be selected
     * if no initial media settings are set. If initial media settings are set this parameter will be ignored. Available options are:
     *
     * MediaController.TRACK_SELECTION_MODE_HIGHEST_BITRATE
     * this mode makes the player select the track with a highest bitrate. This mode is a default mode.
     *
     * MediaController.TRACK_SELECTION_MODE_WIDEST_RANGE
     * this mode makes the player select the track with a widest range of bitrates
     *
     * @param {string} mode
     * @memberof module:MediaPlayer
     * @instance
     */
    function setSelectionModeForInitialTrack(mode) {
        mediaController.setSelectionModeForInitialTrack(mode);
    }

    /**
     * This method returns the track selection mode.
     *
     * @returns {string} mode
     * @memberof module:MediaPlayer
     * @instance
     */
    function getSelectionModeForInitialTrack() {
        return mediaController.getSelectionModeForInitialTrack();
    }

    /**
     * @deprecated since version 2.0 Instead use {@link module:MediaPlayer#getAutoSwitchQualityFor getAutoSwitchQualityFor()}.
     * @returns {boolean} Current state of adaptive bitrate switching
     * @memberof module:MediaPlayer
     * @instance
     */
    function getAutoSwitchQuality() {
        return abrController.getAutoSwitchBitrateFor('video') || abrController.getAutoSwitchBitrateFor('audio');
    }

    /**
     * Set to false to switch off adaptive bitrate switching.
     *
     * @deprecated since version 2.0 Instead use {@link module:MediaPlayer#setAutoSwitchQualityFor setAutoSwitchQualityFor()}.
     * @param {boolean} value
     * @default {boolean} true
     * @memberof module:MediaPlayer
     * @instance
     */
    function setAutoSwitchQuality(value) {
        abrController.setAutoSwitchBitrateFor('video', value);
        abrController.setAutoSwitchBitrateFor('audio', value);
    }

    /**
     * @param {string} type - 'audio' | 'video'
     * @returns {boolean} Current state of adaptive bitrate switching
     * @memberof module:MediaPlayer
     * @instance
     */
    function getAutoSwitchQualityFor(type) {
        return abrController.getAutoSwitchBitrateFor(type);
    }

    /**
     * Set to false to switch off adaptive bitrate switching.
     *
     * @param {string} type - 'audio' | 'video'
     * @param {boolean} value
     * @default {boolean} true
     * @memberof module:MediaPlayer
     * @instance
     */
    function setAutoSwitchQualityFor(type, value) {
        abrController.setAutoSwitchBitrateFor(type, value);
    }


    /**
     * When enabled, after an ABR up-switch in quality, instead of requesting and appending the next fragment
     * at the end of the current buffer range it is requested and appended closer to the current time
     * When enabled, The maximum time to render a higher quality is current time + (1.5 * fragment duration).
     *
     * Note, WHen ABR down-switch is detected, we appended the lower quality at the end of the buffer range to preserve the
     * higher quality media for as long as possible.
     *
     * If enabled, it should be noted there are a few cases when the client will not replace inside buffer range but rather
     * just append at the end.  1. When the buffer level is less than one fragment duration 2.  The client
     * is in an Abandonment State due to recent fragment abandonment event.
     *
     * Known issues:
     * 1. In IE11 with auto switching off, if a user switches to a quality they can not downloaded in time the
     * fragment may be appended in the same range as the playhead or even in past, in IE11 it may cause a stutter
     * or stall in playback.
     *
     *
     * @param {boolean} value
     * @default {boolean} false
     * @memberof module:MediaPlayer
     * @instance
     */
    function setFastSwitchEnabled(value) { //TODO we need to look at track switches for adaptation sets.  If always replace it works much like this but clears buffer. Maybe too many ways to do same thing.
        mediaPlayerModel.setFastSwitchEnabled(value);
    }

    /**
     * @return {boolean} Returns true if FastSwitch ABR is enabled.
     */
    function getFastSwitchEnabled() {
        return mediaPlayerModel.getFastSwitchEnabled();
    }


    /**
     * Enabling buffer-occupancy ABR will switch to the *experimental* implementation of BOLA,
     * replacing the throughput-based ABR rule set (ThroughputRule, BufferOccupancyRule,
     * InsufficientBufferRule and AbandonRequestsRule) with the buffer-occupancy-based
     * BOLA rule set (BolaRule, BolaAbandonRule).
     *
     * @see {@link http://arxiv.org/abs/1601.06748 BOLA WhitePaper.}
     * @see {@link https://github.com/Dash-Industry-Forum/dash.js/wiki/BOLA-status More details about the implementation status.}
     * @param {boolean} value
     * @default false
     * @memberof module:MediaPlayer
     * @instance
     */
    function enableBufferOccupancyABR(value) {
        mediaPlayerModel.setBufferOccupancyABREnabled(value);
    }

    // defaults to false
    function enablerlABR(value) {
        mediaPlayerModel.setrlABREnabled(value);
    }

    /**
     * Allows application to retrieve a manifest.  Manifest loading is asynchro
     * nous and
     * requires the app-provided callback function
     *
     * @param {string} url - url the manifest url
     * @param {function} callback - A Callback function provided when retrieving manifests
     * @memberof module:MediaPlayer
     * @instance
     */
    function retrieveManifest(url, callback) {
        var manifestLoader = createManifestLoader();
        var self = this;

        var handler = function (e) {
            if (!e.error) {
                callback(e.manifest);
            } else {
                callback(null, e.error);
            }
            eventBus.off(Events.INTERNAL_MANIFEST_LOADED, handler, self);
            manifestLoader.reset();
        };

        eventBus.on(Events.INTERNAL_MANIFEST_LOADED, handler, self);

        let uriQueryFragModel = URIQueryAndFragmentModel(context).getInstance();
        uriQueryFragModel.initialize();
        manifestLoader.load(uriQueryFragModel.parseURI(url));
    }

    /**
     * <p>Allows you to set a scheme and server source for UTC live edge detection for dynamic streams.
     * If UTCTiming is defined in the manifest, it will take precedence over any time source manually added.</p>
     * <p>If you have exposed the Date header, use the method {@link module:MediaPlayer#clearDefaultUTCTimingSources clearDefaultUTCTimingSources()}.
     * This will allow the date header on the manifest to be used instead of a time server</p>
     * @param {string} schemeIdUri - <ul>
     * <li>urn:mpeg:dash:utc:http-head:2014</li>
     * <li>urn:mpeg:dash:utc:http-xsdate:2014</li>
     * <li>urn:mpeg:dash:utc:http-iso:2014</li>
     * <li>urn:mpeg:dash:utc:direct:2014</li>
     * </ul>
     * <p>Some specs referencing early ISO23009-1 drafts incorrectly use
     * 2012 in the URI, rather than 2014. support these for now.</p>
     * <ul>
     * <li>urn:mpeg:dash:utc:http-head:2012</li>
     * <li>urn:mpeg:dash:utc:http-xsdate:2012</li>
     * <li>urn:mpeg:dash:utc:http-iso:2012</li>
     * <li>urn:mpeg:dash:utc:direct:2012</li>
     * </ul>
     * @param {string} value - Path to a time source.
     * @default
     * <ul>
     *     <li>schemeIdUri:urn:mpeg:dash:utc:http-xsdate:2014</li>
     *     <li>value:http://time.akamai.com</li>
     * </ul>
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#removeUTCTimingSource removeUTCTimingSource()}
     * @instance
     */
    function addUTCTimingSource(schemeIdUri, value) {
        removeUTCTimingSource(schemeIdUri, value);//check if it already exists and remove if so.
        var vo = new UTCTiming();
        vo.schemeIdUri = schemeIdUri;
        vo.value = value;
        mediaPlayerModel.getUTCTimingSources().push(vo);
    }


    /**
     * <p>Allows you to remove a UTC time source. Both schemeIdUri and value need to match the Dash.vo.UTCTiming properties in order for the
     * entry to be removed from the array</p>
     * @param {string} schemeIdUri - see {@link module:MediaPlayer#addUTCTimingSource addUTCTimingSource()}
     * @param {string} value - see {@link module:MediaPlayer#addUTCTimingSource addUTCTimingSource()}
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#clearDefaultUTCTimingSources clearDefaultUTCTimingSources()}
     * @instance
     */
    function removeUTCTimingSource(schemeIdUri, value) {
        let UTCTimingSources = mediaPlayerModel.getUTCTimingSources();
        UTCTimingSources.forEach(function (obj, idx) {
            if (obj.schemeIdUri === schemeIdUri && obj.value === value) {
                UTCTimingSources.splice(idx, 1);
            }
        });
    }

    /**
     * <p>Allows you to clear the stored array of time sources.</p>
     * <p>Example use: If you have exposed the Date header, calling this method
     * will allow the date header on the manifest to be used instead of the time server.</p>
     * <p>Example use: Calling this method, assuming there is not an exposed date header on the manifest,  will default back
     * to using a binary search to discover the live edge</p>
     *
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#restoreDefaultUTCTimingSources restoreDefaultUTCTimingSources()}
     * @instance
     */
    function clearDefaultUTCTimingSources() {
        mediaPlayerModel.setUTCTimingSources([]);
    }

    /**
     * <p>Allows you to restore the default time sources after calling {@link module:MediaPlayer#clearDefaultUTCTimingSources clearDefaultUTCTimingSources()}</p>
     *
     * @default
     * <ul>
     *     <li>schemeIdUri:urn:mpeg:dash:utc:http-xsdate:2014</li>
     *     <li>value:http://time.akamai.com</li>
     * </ul>
     *
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#addUTCTimingSource addUTCTimingSource()}
     * @instance
     */
    function restoreDefaultUTCTimingSources() {
        addUTCTimingSource(MediaPlayerModel.DEFAULT_UTC_TIMING_SOURCE.scheme, MediaPlayerModel.DEFAULT_UTC_TIMING_SOURCE.value);
    }


    /**
     * <p>Allows you to enable the use of the Date Header, if exposed with CORS, as a timing source for live edge detection. The
     * use of the date header will happen only after the other timing source that take precedence fail or are omitted as described.
     * {@link module:MediaPlayer#clearDefaultUTCTimingSources clearDefaultUTCTimingSources()} </p>
     *
     * @param {boolean} value - true to enable
     * @default {boolean} True
     * @memberof module:MediaPlayer
     * @see {@link module:MediaPlayer#addUTCTimingSource addUTCTimingSource()}
     * @instance
     */
    function enableManifestDateHeaderTimeSource(value) {
        mediaPlayerModel.setUseManifestDateHeaderTimeSource(value);
    }

    /**
     * This value influences the buffer pruning logic.
     * Allows you to modify the buffer that is kept in source buffer in seconds.
     *  0|-----------bufferToPrune-----------|-----bufferToKeep-----|currentTime|
     *
     * @default 30 seconds
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setBufferToKeep(value) {
        mediaPlayerModel.setBufferToKeep(value);
    }

    /**
     * This value influences the buffer pruning logic.
     * Allows you to modify the interval of pruning buffer in seconds.
     *
     * @default 30 seconds
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setBufferPruningInterval(value) {
        mediaPlayerModel.setBufferPruningInterval(value);
    }

    /**
     * The time that the internal buffer target will be set to post startup/seeks (NOT top quality).
     *
     * When the time is set higher than the default you will have to wait longer
     * to see automatic bitrate switches but will have a larger buffer which
     * will increase stability.
     *
     * @default 12 seconds.
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setStableBufferTime(value) {
        mediaPlayerModel.setStableBufferTime(value);
    }

    /**
     * The time that the internal buffer target will be set to once playing the top quality.
     * If there are multiple bitrates in your adaptation, and the media is playing at the highest
     * bitrate, then we try to build a larger buffer at the top quality to increase stability
     * and to maintain media quality.
     *
     * @default 30 seconds.
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setBufferTimeAtTopQuality (value) {
        mediaPlayerModel.setBufferTimeAtTopQuality(value);
    }

    /**
     * The time that the internal buffer target will be set to once playing the top quality for long form content.
     *
     * @default 60 seconds.
     * @see {@link module:MediaPlayer#setLongFormContentDurationThreshold setLongFormContentDurationThreshold()}
     * @see {@link module:MediaPlayer#setBufferTimeAtTopQuality setBufferTimeAtTopQuality()}
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setBufferTimeAtTopQualityLongForm (value) {
        mediaPlayerModel.setBufferTimeAtTopQualityLongForm(value);
    }

    /**
     * The threshold which defines if the media is considered long form content.
     * This will directly affect the buffer targets when playing back at the top quality.
     *
     * @see {@link module:MediaPlayer#setBufferTimeAtTopQualityLongForm setBufferTimeAtTopQualityLongForm()}
     * @default 600 seconds (10 minutes).
     * @param {number} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setLongFormContentDurationThreshold (value) {
        mediaPlayerModel.setLongFormContentDurationThreshold(value);
    }

    /**
     * A threshold, in seconds, of when dashjs abr becomes less conservative since we have a
     * larger "rich" buffer.
     * The BufferOccupancyRule.js rule will override the ThroughputRule's decision when the
     * buffer level surpasses this value and while it remains greater than this value.
     *
     * @default 20 seconds
     * @param {number} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setRichBufferThreshold (value) {
        mediaPlayerModel.setRichBufferThreshold(value);
    }

    /**
     * A percentage between 0.0 and 1 to reduce the measured throughput calculations.
     * The default is 0.9. The lower the value the more conservative and restricted the
     * measured throughput calculations will be. please use carefully. This will directly
     * affect the ABR logic in dash.js
     *
     * @param {number} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setBandwidthSafetyFactor(value) {
        mediaPlayerModel.setBandwidthSafetyFactor(value);
    }

    /**
     * Returns the number of the current BandwidthSafetyFactor
     *
     * @return {number} value
     * @see {@link module:MediaPlayer#setBandwidthSafetyFactor setBandwidthSafetyFactor()}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getBandwidthSafetyFactor() {
        return mediaPlayerModel.getBandwidthSafetyFactor();
    }

    /**
     * A timeout value in seconds, which during the ABRController will block switch-up events.
     * This will only take effect after an abandoned fragment event occurs.
     *
     * @default 10 seconds
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setAbandonLoadTimeout(value) {
        mediaPlayerModel.setAbandonLoadTimeout(value);
    }

    /**
     * Total number of retry attempts that will occur on a fragment load before it fails.
     * Increase this value to a maximum in order to achieve an automatic playback resume
     * in case of completely lost internet connection.
     *
     * @default 3
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setFragmentLoaderRetryAttempts (value) {
        mediaPlayerModel.setFragmentRetryAttempts(value);
    }

    /**
     * Time in milliseconds of which to reload a failed fragment load attempt.
     *
     * @default 1000 milliseconds
     * @param {int} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setFragmentLoaderRetryInterval (value) {
        mediaPlayerModel.setFragmentRetryInterval(value);
    }

    /**
     * Sets whether withCredentials on XHR requests is true or false
     *
     * @default false
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function setXHRWithCredentials(value) {
        mediaPlayerModel.setXHRWithCredentials(value);
    }

    /**
     * Detects if Protection is included and returns an instance of ProtectionController.js
     * @memberof module:MediaPlayer
     * @instance
     */
    function getProtectionController() {
        return detectProtection();
    }

    /**
     * Will override dash.js protection controller.
     * @param {ProtectionController} value - valid protection controller instance.
     * @memberof module:MediaPlayer
     * @instance
     */
    function attachProtectionController(value) {
        protectionController = value;
    }

    /**
     * @param {ProtectionData} value - object containing
     * property names corresponding to key system name strings and associated
     * values being instances of.
     * @memberof module:MediaPlayer
     * @instance
     */
    function setProtectionData(value) {
        protectionData = value;
    }

    /**
     * This method serves to control captions z-index value. If 'true' is passed, the captions will have the highest z-index and be
     * displayed on top of other html elements. Default value is 'false' (z-index is not set).
     * @param {boolean} value
     * @memberof module:MediaPlayer
     * @instance
     */
    function displayCaptionsOnTop(value) {
        let textTracks = TextTracks(context).getInstance();
        textTracks.setConfig({videoModel: videoModel});
        textTracks.initialize();
        textTracks.displayCConTop(value);
    }

    /**
     * Returns instance of Video Container that was attached by calling attachVideoContainer()
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getVideoContainer() {
        return videoModel ? videoModel.getVideoContainer() : null;
    }

    /**
     * Use this method to attach an HTML5 element that wraps the video element.
     *
     * @param {HTMLElement} container - The HTML5 element containing the video element.
     * @memberof module:MediaPlayer
     * @instance
     */
    function attachVideoContainer(container) {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        videoModel.setVideoContainer(container);
    }

    /**
     * Returns instance of Video Element that was attached by calling attachView()
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getVideoElement() {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        return videoModel.getElement();
    }

    /**
     * Use this method to attach an HTML5 VideoElement for dash.js to operate upon.
     *
     * @param {Object} element - An HTMLMediaElement that has already been defined in the DOM (or equivalent stub).
     * @memberof module:MediaPlayer
     * @instance
     */
    function attachView(element) {
        if (!mediaPlayerInitialized) {
            throw MEDIA_PLAYER_NOT_INITIALIZED_ERROR;
        }
        videoModel = null;
        if (element) {
            videoModel = VideoModel(context).getInstance();
            videoModel.initialize();
            videoModel.setElement(element);
            detectProtection();
            detectMetricsReporting();
        }
        resetAndInitializePlayback();
    }

    /**
     * Returns instance of Div that was attached by calling attachTTMLRenderingDiv()
     * @returns {Object}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getTTMLRenderingDiv() {
        return videoModel ? videoModel.getTTMLRenderingDiv() : null;
    }

    /**
     * Use this method to attach an HTML5 div for dash.js to render rich TTML subtitles.
     *
     * @param {HTMLDivElement} div - An unstyled div placed after the video element. It will be styled to match the video size and overlay z-order.
     * @memberof module:MediaPlayer
     * @instance
     */
    function attachTTMLRenderingDiv(div) {
        if (!videoModel) {
            throw ELEMENT_NOT_ATTACHED_ERROR;
        }
        videoModel.setTTMLRenderingDiv(div);
    }

    /**
     * Returns the source string or manifest that was attached by calling attachSource()
     * @returns {string | manifest}
     * @memberof module:MediaPlayer
     * @instance
     */
    function getSource() {
        if (!source) {
            throw SOURCE_NOT_ATTACHED_ERROR;
        }
        return source;
    }

    /**
     * Use this method to set a source URL to a valid MPD manifest file OR
     * a previously downloaded and parsed manifest object.  Optionally, can
     * also provide protection information
     *
     * @param {string|Object} urlOrManifest - A URL to a valid MPD manifest file, or a
     * parsed manifest object.
     *
     *
     * @throws "MediaPlayer not initialized!"
     *
     * @memberof module:MediaPlayer
     * @instance
     */
    function attachSource(urlOrManifest) {
        if (!mediaPlayerInitialized) {
            throw MEDIA_PLAYER_NOT_INITIALIZED_ERROR;
        }

        if (typeof urlOrManifest === 'string') {
            var uriQueryFragModel = URIQueryAndFragmentModel(context).getInstance();
            uriQueryFragModel.initialize();
            source = uriQueryFragModel.parseURI(urlOrManifest);
        } else {
            source = urlOrManifest;
        }

        resetAndInitializePlayback();
    }

    /**
     * Sets the MPD source and the video element to null. You can also reset the MediaPlayer by
     * calling attachSource with a new source file.
     *
     * @memberof module:MediaPlayer
     * @instance
     */
    function reset() {
        attachSource(null);
        attachView(null);
        protectionData = null;
        protectionController = null;
    }

    //***********************************
    // PRIVATE METHODS
    //***********************************

    function resetAndInitializePlayback() {
        if (playbackInitialized) {
            playbackInitialized = false;
            adapter.reset();
            streamController.reset();
            playbackController.reset();
            abrController.reset();
            rulesController.reset();
            mediaController.reset();
            streamController = null;
            metricsReportingController = null;
            if (isReady()) {
                initializePlayback();
            }
        } else if (isReady()) {
            initializePlayback();
        }
    }

    function createControllers() {

        let synchronizationRulesCollection = SynchronizationRulesCollection(context).getInstance();
        synchronizationRulesCollection.initialize();

        let abrRulesCollection = ABRRulesCollection(context).getInstance();
        abrRulesCollection.initialize();

        //let scheduleRulesCollection = ScheduleRulesCollection(context).getInstance();
        //scheduleRulesCollection.initialize();

        let sourceBufferController = SourceBufferController(context).getInstance();
        sourceBufferController.setConfig({dashManifestModel: dashManifestModel});

        mediaController.initialize();
        mediaController.setConfig({
            errHandler: errHandler
        });

        rulesController = RulesController(context).getInstance();
        rulesController.initialize();
        rulesController.setConfig({
            abrRulesCollection: abrRulesCollection,
            synchronizationRulesCollection: synchronizationRulesCollection
        });

        streamController = StreamController(context).getInstance();
        streamController.setConfig({
            capabilities: capabilities,
            manifestLoader: createManifestLoader(),
            manifestModel: ManifestModel(context).getInstance(),
            dashManifestModel: dashManifestModel,
            protectionController: protectionController,
            adapter: adapter,
            metricsModel: metricsModel,
            dashMetrics: dashMetrics,
            liveEdgeFinder: LiveEdgeFinder(context).getInstance(),
            mediaSourceController: MediaSourceController(context).getInstance(),
            timeSyncController: TimeSyncController(context).getInstance(),
            baseURLController: BaseURLController(context).getInstance(),
            errHandler: errHandler,
            timelineConverter: TimelineConverter(context).getInstance()
        });
        streamController.initialize(autoPlay, protectionData);

        abrController.setConfig({
            abrRulesCollection: abrRulesCollection,
            rulesController: rulesController,
            streamController: streamController
        });
    }

    function createManifestLoader() {
        return ManifestLoader(context).create({
            errHandler: errHandler,
            parser: createManifestParser(),
            metricsModel: metricsModel,
            requestModifier: RequestModifier(context).getInstance()
        });
    }

    function createManifestParser() {
        //TODO-Refactor Need to be able to switch this create out so will need API to set which parser to use?
        return DashParser(context).create();
    }

    function createAdaptor() {
        //TODO-Refactor Need to be able to switch this create out so will need API to set which adapter to use? Handler is created is inside streamProcessor so need to figure that out as well
        adapter = DashAdapter(context).getInstance();
        adapter.initialize();
        adapter.setConfig({dashManifestModel: dashManifestModel});
        return adapter;
    }

    function detectProtection() {
        if (protectionController) {
            return protectionController;
        }
        // do not require Protection as dependencies as this is optional and intended to be loaded separately
        let Protection = dashjs.Protection; /* jshint ignore:line */
        if (typeof Protection === 'function') {//TODO need a better way to register/detect plugin components
            let protection = Protection(context).create();
            Events.extend(Protection.events);
            MediaPlayerEvents.extend(Protection.events, { publicOnly: true });
            protectionController = protection.createProtectionSystem({
                log: log,
                videoModel: videoModel,
                capabilities: capabilities,
                eventBus: eventBus,
                adapter: adapter
            });
            return protectionController;
        }

        return null;
    }

    function detectMetricsReporting() {
        if (metricsReportingController) {
            return metricsReportingController;
        }
        // do not require MetricsReporting as dependencies as this is optional and intended to be loaded separately
        let MetricsReporting = dashjs.MetricsReporting; /* jshint ignore:line */
        if (typeof MetricsReporting === 'function') {//TODO need a better way to register/detect plugin components
            let metricsReporting = MetricsReporting(context).create();

            metricsReportingController = metricsReporting.createMetricsReporting({
                log: log,
                eventBus: eventBus,
                mediaElement: getVideoElement(),
                dashManifestModel: dashManifestModel,
                metricsModel: metricsModel
            });

            return metricsReportingController;
        }

        return null;
    }

    function getDVRInfoMetric() {
        var metric = metricsModel.getReadOnlyMetricsFor('video') || metricsModel.getReadOnlyMetricsFor('audio');
        return dashMetrics.getCurrentDVRInfo(metric);
    }

    function getAsUTC(valToConvert) {
        var metric = getDVRInfoMetric();
        var availableFrom,
            utcValue;

        if (!metric) {
            return 0;
        }
        availableFrom = metric.manifestInfo.availableFrom.getTime() / 1000;
        utcValue = valToConvert + (availableFrom + metric.range.start);
        return utcValue;
    }

    function getActiveStream() {
        if (!playbackInitialized) {
            throw PLAYBACK_NOT_INITIALIZED_ERROR;
        }
        var streamInfo = streamController.getActiveStreamInfo();
        return streamInfo ? streamController.getStreamById(streamInfo.id) : null;
    }

    function initializePlayback() {
        if (!playbackInitialized) {
            playbackInitialized = true;
            log('Playback Initialized');
            createControllers();
            if (typeof source === 'string') {
                streamController.load(source);
            } else {
                streamController.loadWithManifest(source);
            }
        }
    }

    instance = {
        initialize: initialize,
        on: on,
        off: off,
        extend: extend,
        attachView: attachView,
        attachSource: attachSource,
        isReady: isReady,
        play: play,
        isPaused: isPaused,
        pause: pause,
        isSeeking: isSeeking,
        seek: seek,
        setMute: setMute,
        isMuted: isMuted,
        setVolume: setVolume,
        getVolume: getVolume,
        setAbrAlgorithm: setAbrAlgorithm,
        time: time,
        duration: duration,
        timeAsUTC: timeAsUTC,
        durationAsUTC: durationAsUTC,
        getActiveStream: getActiveStream,
        getDVRWindowSize: getDVRWindowSize,
        getDVRSeekOffset: getDVRSeekOffset,
        convertToTimeCode: convertToTimeCode,
        formatUTC: formatUTC,
        getVersion: getVersion,
        getDebug: getDebug,
        getBufferLength: getBufferLength,
        getVideoModel: getVideoModel,
        getVideoContainer: getVideoContainer,
        getTTMLRenderingDiv: getTTMLRenderingDiv,
        getVideoElement: getVideoElement,
        getSource: getSource,
        setLiveDelayFragmentCount: setLiveDelayFragmentCount,
        setLiveDelay: setLiveDelay,
        useSuggestedPresentationDelay: useSuggestedPresentationDelay,
        enableLastBitrateCaching: enableLastBitrateCaching,
        enableLastMediaSettingsCaching: enableLastMediaSettingsCaching,
        setMaxAllowedBitrateFor: setMaxAllowedBitrateFor,
        getMaxAllowedBitrateFor: getMaxAllowedBitrateFor,
        setMaxAllowedRepresentationRatioFor: setMaxAllowedRepresentationRatioFor,
        getMaxAllowedRepresentationRatioFor: getMaxAllowedRepresentationRatioFor,
        setAutoPlay: setAutoPlay,
        getAutoPlay: getAutoPlay,
        setScheduleWhilePaused: setScheduleWhilePaused,
        getScheduleWhilePaused: getScheduleWhilePaused,
        getDashMetrics: getDashMetrics,
        getMetricsFor: getMetricsFor,
        getQualityFor: getQualityFor,
        setQualityFor: setQualityFor,
        getLimitBitrateByPortal: getLimitBitrateByPortal,
        setLimitBitrateByPortal: setLimitBitrateByPortal,
        getUsePixelRatioInLimitBitrateByPortal: getUsePixelRatioInLimitBitrateByPortal,
        setUsePixelRatioInLimitBitrateByPortal: setUsePixelRatioInLimitBitrateByPortal,
        setTextTrack: setTextTrack,
        getBitrateInfoListFor: getBitrateInfoListFor,
        setInitialBitrateFor: setInitialBitrateFor,
        getInitialBitrateFor: getInitialBitrateFor,
        setInitialRepresentationRatioFor: setInitialRepresentationRatioFor,
        getInitialRepresentationRatioFor: getInitialRepresentationRatioFor,
        getStreamsFromManifest: getStreamsFromManifest,
        getTracksFor: getTracksFor,
        getTracksForTypeFromManifest: getTracksForTypeFromManifest,
        getCurrentTrackFor: getCurrentTrackFor,
        setInitialMediaSettingsFor: setInitialMediaSettingsFor,
        getInitialMediaSettingsFor: getInitialMediaSettingsFor,
        setCurrentTrack: setCurrentTrack,
        getTrackSwitchModeFor: getTrackSwitchModeFor,
        setTrackSwitchModeFor: setTrackSwitchModeFor,
        setSelectionModeForInitialTrack: setSelectionModeForInitialTrack,
        getSelectionModeForInitialTrack: getSelectionModeForInitialTrack,
        getAutoSwitchQuality: getAutoSwitchQuality,
        setAutoSwitchQuality: setAutoSwitchQuality,
        setFastSwitchEnabled: setFastSwitchEnabled,
        getFastSwitchEnabled: getFastSwitchEnabled,
        getAutoSwitchQualityFor: getAutoSwitchQualityFor,
        setAutoSwitchQualityFor: setAutoSwitchQualityFor,
        enableBufferOccupancyABR: enableBufferOccupancyABR,
        enablerlABR: enablerlABR,
        setBandwidthSafetyFactor: setBandwidthSafetyFactor,
        getBandwidthSafetyFactor: getBandwidthSafetyFactor,
        setAbandonLoadTimeout: setAbandonLoadTimeout,
        retrieveManifest: retrieveManifest,
        addUTCTimingSource: addUTCTimingSource,
        removeUTCTimingSource: removeUTCTimingSource,
        clearDefaultUTCTimingSources: clearDefaultUTCTimingSources,
        restoreDefaultUTCTimingSources: restoreDefaultUTCTimingSources,
        setBufferToKeep: setBufferToKeep,
        setBufferPruningInterval: setBufferPruningInterval,
        setStableBufferTime: setStableBufferTime,
        setBufferTimeAtTopQuality: setBufferTimeAtTopQuality,
        setFragmentLoaderRetryAttempts: setFragmentLoaderRetryAttempts,
        setFragmentLoaderRetryInterval: setFragmentLoaderRetryInterval,
        setXHRWithCredentials: setXHRWithCredentials,
        setBufferTimeAtTopQualityLongForm: setBufferTimeAtTopQualityLongForm,
        setLongFormContentDurationThreshold: setLongFormContentDurationThreshold,
        setRichBufferThreshold: setRichBufferThreshold,
        getProtectionController: getProtectionController,
        attachProtectionController: attachProtectionController,
        setProtectionData: setProtectionData,
        enableManifestDateHeaderTimeSource: enableManifestDateHeaderTimeSource,
        displayCaptionsOnTop: displayCaptionsOnTop,
        attachVideoContainer: attachVideoContainer,
        attachTTMLRenderingDiv: attachTTMLRenderingDiv,
        reset: reset,
        isInitialized: isInitialized
    };

    setup();

    return instance;
}

MediaPlayer.__dashjs_factory_name = 'MediaPlayer';
let factory = FactoryMaker.getClassFactory(MediaPlayer);
factory.events = MediaPlayerEvents;
export default factory;
