CREATE TABLE Users (
    user_id VARCHAR(20) PRIMARY KEY,
    pwd VARCHAR(20)  ,
    is_admin BOOLEAN
);


CREATE TABLE Items (
    item_id SERIAL PRIMARY KEY,
    category INT,
    description VARCHAR(50),
    condition INT,
    seller_id VARCHAR(20),
    buy_it_now_price NUMERIC(10, 2) ,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    bid_closing_date TIMESTAMPTZ ,
    FOREIGN KEY (seller_id) REFERENCES Users(user_id),
    CONSTRAINT category_check CHECK (category BETWEEN 1 AND 7),
    CONSTRAINT condition_check CHECK (condition BETWEEN 1 AND 4)
);


CREATE TABLE Bids (
    bid_id SERIAL PRIMARY KEY,
    item_id INT  ,
    bid_price NUMERIC(10, 2)  ,
    bidder_id VARCHAR(20)  ,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES Items(item_id),
    FOREIGN KEY (bidder_id) REFERENCES Users(user_id)
);


CREATE TABLE OldBids (
    bid_id SERIAL PRIMARY KEY,
    item_id INT  ,
    bid_price NUMERIC(10, 2)  ,
    bidder_id VARCHAR(20)  ,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES Items(item_id),
    FOREIGN KEY (bidder_id) REFERENCES Users(user_id)
);


CREATE TABLE Billing (
    transaction_id SERIAL PRIMARY KEY,
    item_id INT  ,
    sold_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    seller_id VARCHAR(20)  ,
    buyer_id VARCHAR(20)  ,
    price NUMERIC(10, 2)  ,
    FOREIGN KEY (item_id) REFERENCES Items(item_id),
    FOREIGN KEY (seller_id) REFERENCES Users(user_id),
    FOREIGN KEY (buyer_id) REFERENCES Users(user_id)
);