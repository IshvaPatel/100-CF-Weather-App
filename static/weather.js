document.addEventListener('DOMContentLoaded', () => {
    loadFavorites();
});

async function fetchWeather() {
    console.log("Hello")
    const city = document.getElementById('city').value;
    const weatherInfoElement = document.getElementById('weather-info');
    const weatherInfoElement2 = document.getElementById('weather-info2');
    const temperatureElement = document.getElementById('temperature');
    const humidityElement = document.getElementById('humidity');
    const windSpeedElement = document.getElementById('wind-speed');
    weatherInfoElement.style.display = 'block';
    weatherInfoElement2.style.display = 'block';
    weatherInfoElement.innerText = 'Loading...';

    const weatherApiKey1 = '7dfcf22c94fa8abff845e481a769bef4'; // Your OpenWeatherMap API key
    const weatherApiKey = '105c37a85acf04c0265486c669d39389';
    const geoencodingUrl = `http://api.openweathermap.org/geo/1.0/direct?q=${city}&limit=1&appid=${weatherApiKey1}`;

    try {
        const geoencodingResponse = await fetch(geoencodingUrl);
        if (!geoencodingResponse.ok) {
            throw new Error('Geocoding request failed: ' + geoencodingResponse.statusText);
        }
        const geoencodingData = await geoencodingResponse.json();
        if (!geoencodingData.length) {
            throw new Error('Geocoding error: No results found');
        }
        const { lat, lon } = geoencodingData[0] ;

        const weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${weatherApiKey}`;
        const weatherResponse = await fetch(weatherUrl);
        if (!weatherResponse.ok) {
            throw new Error('Weather request failed: ' + weatherResponse.statusText);
        }
        const weatherData = await weatherResponse.json();
        const currentWeather = weatherData.current;
        const temperature = weatherData.main.temp;
        const humidity = weatherData.main.humidity;
        const windSpeed = weatherData.wind.speed;
        const weatherDescription = weatherData.weather[0].description;
        const weatherIcon = weatherData.weather[0].icon;
        temperatureElement.innerText = `${temperature} K`;
        humidityElement.innerText = `${humidity} %`;
        windSpeedElement.innerText = `${windSpeed} m/s`;
        weatherInfoElement2.style.display = 'block';
        weatherInfoElement.innerText = `Description: ${weatherDescription}`;
    } catch (error) {
        weatherInfoElement.innerText = `Error: ${error.message}`;
        weatherInfoElement.style.display = 'block'; // Show the error message
    }
}

function saveFavorite() {
    const city = document.getElementById('city').value;
    
    if (!city) {
        alert('Please enter a city name to save as favorite.');
        return;
    }

    let favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    if (!favorites.includes(city)) {
        favorites.push(city);
        localStorage.setItem('favorites', JSON.stringify(favorites));
        loadFavorites();
    } else {
        alert('City is already in your favorite searches.');
    }
}

function loadFavorites() {
    const favoritesListElement = document.getElementById('favorites-list');
    favoritesListElement.innerHTML = '';

    let favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    favorites.forEach(city => {
        let listItem = document.createElement('li');
        listItem.innerText = city;
        listItem.onclick = () => {
            document.getElementById('city').value = city;
            fetchWeather();
        };
        favoritesListElement.appendChild(listItem);
    });
}