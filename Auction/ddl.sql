CREATE TABLE Users (
    user_id VARCHAR(20) PRIMARY KEY,
    pwd VARCHAR(20) NOT NULL,
    is_admin BOOLEAN
);


CREATE TABLE Items (
    item_id SERIAL PRIMARY KEY,
    category INT NOT NULL,
    description VARCHAR(50),
    condition INT NOT NULL,
    seller_id VARCHAR(20) NOT NULL,
    buy_it_now_price NUMERIC(10, 2) NOT NULL,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    bid_closing_date TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (seller_id) REFERENCES Users(user_id),
    CONSTRAINT category_check CHECK (category BETWEEN 1 AND 7),
    CONSTRAINT condition_check CHECK (condition BETWEEN 1 AND 4)
);


CREATE TABLE Bids (
    bid_id SERIAL PRIMARY KEY,
    item_id INT NOT NULL,
    bid_price NUMERIC(10, 2) NOT NULL,
    bidder_id VARCHAR(20) NOT NULL,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    bid_closing_date TIMESTAMPTZ,
    FOREIGN KEY (item_id, bid_closing_date) REFERENCES Items(item_id, bid_closing_date),
    FOREIGN KEY (bidder_id) REFERENCES Users(user_id)
);


CREATE TABLE OldBids (
    bid_id SERIAL PRIMARY KEY,
    item_id INT NOT NULL,
    bid_price NUMERIC(10, 2) NOT NULL,
    bidder_id VARCHAR(20) NOT NULL,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    bid_closing_date TIMESTAMPTZ,
    FOREIGN KEY (item_id, bid_closing_date) REFERENCES Items(item_id, bid_closing_date),
    FOREIGN KEY (bidder_id) REFERENCES Users(user_id)
);


CREATE TABLE Billing (
    transaction_id SERIAL PRIMARY KEY,
    item_id INT NOT NULL,
    sold_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    seller_id VARCHAR(20) NOT NULL,
    buyer_id VARCHAR(20) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    FOREIGN KEY (item_id) REFERENCES Items(item_id),
    FOREIGN KEY (seller_id) REFERENCES Users(user_id),
    FOREIGN KEY (buyer_id) REFERENCES Users(user_id)
);