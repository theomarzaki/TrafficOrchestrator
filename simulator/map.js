/* eslint-disable */
const icons = [
    "./images/number_1.png",
    "./images/number_2.png",
    "./images/number_3.png",
    "./images/number_4.png",
    "./images/number_5.png",
    "./images/number_6.png",
    "./images/number_7.png",
    "./images/number_8.png",
    "./images/number_9.png",
];

const size = new OpenLayers.Size(21, 25);
const offset = new OpenLayers.Pixel(-(size.w / 2), -size.h);

let data;
const MANTISSA_DIVIDER = 10000000;
const {map, markers} = initMap("mapdiv");
let ruCount = 0;
let markersCache = [];


function initMap(mapDivId) {
    const map = new OpenLayers.Map(mapDivId);
    map.addLayer(new OpenLayers.Layer.OSM());
    const center = new OpenLayers.LonLat(2.2432911, 48.6252673).transform(
        new OpenLayers.Projection("EPSG:4326"), // transform from WGS 1984
        map.getProjectionObject() // to Spherical Mercator Projection
    );
    const markers = new OpenLayers.Layer.Markers("Markers");
    map.addLayer(markers);

    const zoom = 17;

    map.setCenter(center, zoom);
    return {
        map,
        markers
    };
}

function createMarker(longitude, latitude, markerNumber) {
    let icon;
    if (markerNumber === undefined) {
        icon = new OpenLayers.Icon('./image/marker.png', size, offset);
    } else {
        icon = new OpenLayers.Icon(icons[markerNumber], size, offset);
    }
    const lonLat = new OpenLayers.LonLat(longitude, latitude).transform(
        new OpenLayers.Projection("EPSG:4326"), // transform from WGS 1984
        map.getProjectionObject() // to Spherical Mercator Projection
    );
    const marker = new OpenLayers.Marker(lonLat, icon);
    return marker;
}


function loadFile(fileName) {
    let json = null;
    $.ajax({
        'async': false,
        'global': false,
        'url': fileName,
        'dataType': "json",
        'success': (results) => {
            json = results;
        }
    });
    console.log(json);
    return json;
}

function readLoop(index) {
    setTimeout(() => {
        let found = false;
        const ruId = data[index].message.ru_description_list[0].message.uuid;
        for (const object of markersCache) {
            if (object.ruId === ruId) {
                found = true;
                markers.removeMarker(object.marker);
                object.marker = createMarker(data[index].message.ru_description_list[0].message.position.longitude / MANTISSA_DIVIDER,
                    data[index].message.ru_description_list[0].message.position.latitude / MANTISSA_DIVIDER, object.iconNumber);
                markers.addMarker(object.marker);
            }
        }
        if (!found) {
            ruCount++;
            marker = createMarker(data[index].message.ru_description_list[0].message.position.longitude / MANTISSA_DIVIDER,
                data[index].message.ru_description_list[0].message.position.latitude / MANTISSA_DIVIDER, ruCount);
            markersCache.push({
                ruId,
                marker,
                iconNumber: ruCount
            });
            markers.addMarker(marker);

            console.log(`icon ${ruCount} is RU with id ${ruId}`);
            addInfo(`icon ${ruCount} is RU with id ${ruId}`);
        }
        if (index % 1000 === 0) {
            console.log(`index is at ${index}`);
        }
        if (data.length > index + 1) {
            readLoop(index + 1);
        } else {
            console.log(`file ended after ${index} iterations`);
        }
    }, 20);
}

function compare(rudA, rudB) {
    if (rudA.timestamp < rudB.timestamp) {
        return -1;
    }
    if (rudA.timestamp > rudB.timestamp) {
        return 1;
    }
    return 0;
}

function addInfo(text) {
    const div = document.createElement('div');

    div.className = 'row';

    div.innerHTML = text;
    document.getElementById('info').
        appendChild(div);
}

function onStart() {
    markers.clearMarkers();
    ruCount = 0;
    markersCache = [];

    document.getElementById('info').innerHTML = '<h2>Info</h2>';
    console.log("loading file /data/" + $("#fileName").
        val());
    // move center to vigo if using vigo file
    if ($("#fileName").val().indexOf("vigo")!== -1) {
        const center = new OpenLayers.LonLat(-8.6137971, 42.1027403).transform(
            new OpenLayers.Projection("EPSG:4326"), // transform from WGS 1984
            map.getProjectionObject() // to Spherical Mercator Projection
        );
        map.setCenter(center, 17);
    } else {
        const center = new OpenLayers.LonLat(2.2432911, 48.6252673).transform(
            new OpenLayers.Projection("EPSG:4326"), // transform from WGS 1984
            map.getProjectionObject() // to Spherical Mercator Projection
        );
        map.setCenter(center, 17);
    }
    data = loadFile("data/" + $("#fileName").val());
    readLoop(0);
}
